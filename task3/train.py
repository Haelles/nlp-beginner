import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext import data
import numpy as np
import spacy
import math

import time
from datetime import datetime

import sys
sys.path.append(".")
import os
from utils.iterators import build_iterator
from model.ESIM import ESIM


def train(args):
    cur_date_time = datetime.strftime(datetime.now(),'%Y-%m-%d-%H_%M_%S')
    exp_name = str(cur_date_time)
    writer = SummaryWriter('runs/' + exp_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # data path
    train_file_path = args.data + 'snli_1.0_train.jsonl'
    val_file_path = args.data + 'snli_1.0_dev.jsonl'
    
    # load hyper parameter
    epoch = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    gradient_clip = args.gradient_clip
    embedded_dim = args.embed
    hidden_size = args.hidden
    patience = args.patience

    vector_file_name = args.vector

    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True)

    train_iterator = build_iterator(train_file_path, text_field=text_field, batch_size=batch_size, device=device, is_train=True, vector_file_name=vector_file_name)
    val_iterator = build_iterator(val_file_path, text_field=text_field, batch_size=batch_size, device=device, is_train=False, vector_file_name=vector_file_name)

    model = ESIM(text_field, embedded_dim=embedded_dim, hidden_size=hidden_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO 试试效果怎样
    # val过拟合
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)
    criterion = nn.CrossEntropyLoss()

    cur_iteration = 0
    best_acc = -1.0
    patience_counter = 0
    iters_per_epoch = math.ceil(len(train_iterator.dataset) * 1.0 / batch_size)
    for cur_epoch in range(epoch):
        model.train()
        
        start_time = time.time()

        train_total_correct = 0  # 每个epoch统计这个epoch总的正确率
        train_total = 0  # 计算一共有多少数据
        total_loss = 0.0 # 用于统计每个epoch总的平均loss

        for idx, batch in enumerate(train_iterator):
            # prepare data
            premise, premise_length = batch.premise
            premise_length = premise_length.to('cpu').int()
            premise = premise.transpose(0, 1).contiguous()
            hypothesis, hypothesis_length  = batch.hypothesis
            hypothesis = hypothesis.transpose(0, 1).contiguous()  # should be [b, l]
            hypothesis_length = hypothesis_length.to('cpu').int()
            label = batch.label  # [batch] e.g. torch.Size([32])

            optimizer.zero_grad()

            logits, prediction = model(premise, premise_length, hypothesis, hypothesis_length)
            loss = criterion(logits, label)  # logits should be [b, 3], label should be [b]
            
            loss.backward()
            # 裁剪梯度
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            
            train_result = torch.max(logits, dim=-1)[1]  # should be [b]
            correct = (train_result == label).sum()  # tensor scalar
            train_total_correct += correct.item()
            train_total += logits.shape[0]
            total_loss += loss.item()

            if cur_iteration % 500 == 0:
                writer.add_scalar('training loss', loss.item(), cur_iteration + 1)
                print('training epoch {epoch} total_iteration {iter} loss:{loss}'.format(epoch=cur_epoch + 1, iter=cur_iteration + 1, loss=loss.item()))

            cur_iteration += 1

        print('Training epoch {epoch} done. Time used: {time:.2f} mins'.format(epoch=cur_epoch + 1, time=(time.time()-start_time) / 60.0))
        epoch_avg_acc = train_total_correct * 1.0 / train_total
        epoch_avg_loss = total_loss / iters_per_epoch
        print('total_data added: {train_total} len(iterator.dataset): {len}'.format(train_total=train_total, len=len(train_iterator.dataset)))
        print('epoch_avg_acc {epoch_avg_acc:.4f} epoch_avg_loss {epoch_avg_loss}'.format(epoch_avg_acc=epoch_avg_acc, epoch_avg_loss=epoch_avg_loss))
        writer.add_scalar('epoch_avg_acc', epoch_avg_acc, cur_epoch + 1)
        writer.add_scalar('epoch_avg_loss', epoch_avg_loss, cur_epoch + 1)

        # begin to val
        model.eval()

        val_start_time = time.time()
        val_total_correct = 0
        val_total_data = len(val_iterator.dataset)  # 9842
        val_total_loss = 0.0
        val_step = 0
        val_acc = 0.0
        with torch.no_grad():
            for idx, val_batch in enumerate(val_iterator):
                # prepare val data
                val_premise, val_premise_length = batch.premise
                val_premise = val_premise.transpose(0, 1).contiguous()
                val_premise_length = val_premise_length.to('cpu').int()  # 文档要求用CPU int64 tensor
                val_hypothesis, val_hypothesis_length  = batch.hypothesis
                val_hypothesis = val_hypothesis.transpose(0, 1).contiguous()  # should be [b, l]
                val_hypothesis_length = val_hypothesis_length.to('cpu').int()  # 文档要求用CPU int64 tensor
                val_label = batch.label  # [batch] e.g. torch.Size([32])

                val_logits, val_prediction = model(val_premise, val_premise_length, val_hypothesis, val_hypothesis_length)
                loss = criterion(val_logits, val_label)
                val_total_loss += loss.item()
                val_result = torch.max(val_logits, dim=-1)[1]
                val_correct = (val_result == val_label).sum()
                val_total_correct += val_correct.item()
                val_step += 1
            
            val_loss_avg = val_total_loss / val_step
            print('iters step cnt:{step} math.ceil:{ceil}'.format(step=val_step, ceil=math.ceil(val_total_data / batch_size)))
            val_acc = 100 * (val_total_correct * 1.0 / val_total_data)
            
            print('Val {epoch} avg_loss: {loss} acc: {acc}%'.format(epoch=cur_epoch + 1, loss=val_loss_avg, acc=val_acc))
            writer.add_scalar('val loss per epoch', val_loss_avg, cur_epoch + 1)
            writer.add_scalar('val acc per epoch', val_acc, cur_epoch + 1)

        print('val epoch {epoch} done. Val time used: {time} mins'.format(epoch=cur_epoch + 1, time=(time.time()-val_start_time) / 60.0))
        
        # # update lr
        # scheduler.step(val_acc)

        # early stopping和存储最优ckp
        if val_acc < best_acc:
            patience_counter += 1  # 累计一次，当前epoch性能不如前一个epoch
        else:
            best_acc = val_acc
            patience_counter = 0  # 清零
        if not os.path.exists(os.path.join('checkpoints/ESIM/', exp_name)):
            os.makedirs(os.path.join('checkpoints/ESIM/', exp_name))
        torch.save(model, 'checkpoints/ESIM/{exp_name}/epoch_{epoch}_acc_{acc}'.format(exp_name=exp_name, epoch=cur_epoch + 1, acc=best_acc))
        print('ckp saved in checkpoints/ESIM/{exp_name}/epoch_{epoch}_acc_{acc}'.format(exp_name=exp_name, epoch=cur_epoch + 1, acc=best_acc))
        if patience_counter >= patience:
            print('现结束了第{epoch}次训练和验证 连续{patience}次性能下降 训练停止'.format(epoch=cur_epoch + 1, patience=patience))
            break

    writer.close()  # MUST ADD THIS!
    
    print('Total {iter} iterations. Total {epoch} epoches'.format(iter=cur_iteration, epoch=epoch))
    print('training process done')


def generate_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--epoch', type=int, default=64, help='training epochs')
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--gradient_clip', type=float, default=10.0, help='max norm for gradient clipping')
    parser.add_argument('--embed', type=int, default=300, help='input size')
    parser.add_argument('--hidden', type=int, default=300, help='hidden size')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')

    parser.add_argument('--data', type=str, default='../data/snli_1.0/', help='dataset path')
    parser.add_argument('--vector', type=str, default='glove.840B.300d.txt', help='vector file name')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = generate_args()
    train(args)