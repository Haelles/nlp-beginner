import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext import data
import numpy as np
import spacy

import time
from datetime import datetime

from utils.iterators import build_iterator
from model.TextCNN import TextCNN
from model.TextRNN import TextRNN


def train(args):
    cur_date_time = datetime.strftime(datetime.now(),'%Y-%m-%d-%H:%M:%S')
    writer = SummaryWriter('runs/' + str(cur_date_time))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_file_path = args.data + 'train.csv'
    val_file_path = args.data + 'val.csv'
    train_tag = 'train'    
    batch_size = args.batch_size
    epoch = args.epoch
    
    lr = args.lr

    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)

    assert train_tag in ['train', 'val', 'test']

    train_iterator = build_iterator(train_file_path, text_field=text_field, batch_size=batch_size, device=device, train_tag=train_tag)
    val_iterator = build_iterator(val_file_path, text_field=text_field, batch_size=batch_size, device=device, train_tag='val')

    if args.model == 'cnn':
       model = TextCNN(text_field=text_field).to(device)
    else:
        model = TextRNN(text_field=text_field).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    cur_iteration = 0
    best_acc = 0.0
    for cur_epoch in range(epoch):
        model.train()
        
        start_time = time.time()

        for idx, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.Sentiment)
            loss.backward()
            optimizer.step()
            
            if cur_iteration % 1000 == 0:
                writer.add_scalar('training loss', loss.item(), cur_iteration + 1)
                print('training epoch {epoch} total_iteration {iter} loss:{loss}'.format(epoch=cur_epoch, iter=cur_iteration, loss=loss.item()))

            cur_iteration += 1

        print('training epoch {epoch} done. Time used: {time} seconds'.format(epoch=cur_epoch, time=time.time()-start_time))
        # begin to val
        model.eval()

        val_start_time = time.time()
        total_correct = 0
        total_val_data = len(val_iterator.dataset)
        total_train_data = len(train_iterator.dataset)
        total_loss = 0.0
        val_step = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for idx, val_batch in enumerate(val_iterator):
                val_logits = model(val_batch)
                loss = criterion(val_logits, val_batch.Sentiment)
                total_loss += loss.item()
                val_result = torch.max(val_logits, dim=-1)[1]
                correct = (val_result == val_batch.Sentiment).sum()
                total_correct += correct.item()
                val_step += 1.0
            
            val_loss_avg = total_loss / val_step
            val_acc = 100 * (total_correct * 1.0 / total_val_data)
            
            print('Val {epoch} loss: {loss} acc: {acc}%'.format(epoch=cur_epoch, loss=val_loss_avg, acc=val_acc))
            writer.add_scalar('val loss per epoch', val_loss_avg, cur_epoch + 1)
            writer.add_scalar('val acc per epoch', val_acc, cur_epoch + 1)
            print('draw {epoch}'.format(epoch=cur_epoch))

        print('val epoch {epoch} done. Val time used: {time} seconds'.format(epoch=cur_epoch, time=time.time()-val_start_time))
        if best_acc < val_acc:
            best_acc = val_acc
            date_time = str(datetime.strftime(datetime.now(),'%Y-%m-%d-%H:%M:%S'))
            torch.save(model, 'checkpoints/model_{model}_epoch_{epoch}_acc_{acc}_{date}'.format(model=args.model, epoch=cur_epoch, acc=best_acc, date=date_time))
            print('ckp saved in checkpoints/model_{model}_epoch_{epoch}_acc_{acc}_{date}'.format(model=args.model, epoch=cur_epoch, acc=best_acc, date=date_time))

    writer.close()  # MUST ADD THIS!
    
    print('total_iterations {iter}'.format(iter=cur_iteration))
    print('training process done')


def generate_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--epoch', type=int, default=4, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'rnn'],help='the model used for training')

    parser.add_argument('--data', type=str, default='data/sentiment-analysis-on-movie-reviews/', help='dataset path')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = generate_args()
    train(args)