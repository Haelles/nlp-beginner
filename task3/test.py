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
from utils.iterators import build_iterator
from model.ESIM import ESIM


def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # data path
    test_file_path = args.data + 'snli_1.0_test.jsonl'
    
    # load parameter
    batch_size = args.batch_size
    vector_file_name = args.vector
    load_from = args.load_from

    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True)

    test_iterator = build_iterator(test_file_path, text_field=text_field, batch_size=batch_size, device=device, is_train=False, vector_file_name=vector_file_name)

    model = torch.load(load_from).to(device)
    criterion = nn.CrossEntropyLoss()
    # begin to test
    model.eval()

    test_start_time = time.time()
    test_total_correct = 0
    test_total_data = len(test_iterator.dataset)
    test_total_loss = 0.0
    test_step = math.ceil(test_total_data / batch_size)
    test_acc = 0.0
    with torch.no_grad():
        for idx, test_batch in enumerate(test_iterator):
            # prepare test data
            test_premise, test_premise_length = test_batch.premise
            test_premise = test_premise.transpose(0, 1).contiguous()
            test_premise_length = test_premise_length.to('cpu').int()  # 文档要求用CPU int64 tensor
            test_hypothesis, test_hypothesis_length  = test_batch.hypothesis
            test_hypothesis = test_hypothesis.transpose(0, 1).contiguous()  # should be [b, l]
            test_hypothesis_length = test_hypothesis_length.to('cpu').int()  # 文档要求用CPU int64 tensor
            test_label = test_batch.label  # [batch] e.g. torch.Size([32])

            test_logits, test_prediction = model(test_premise, test_premise_length, test_hypothesis, test_hypothesis_length)
            loss = criterion(test_logits, test_label)
            test_total_loss += loss.item()
            test_result = torch.max(test_logits, dim=-1)[1]
            test_correct = (test_result == test_label).sum()
            test_total_correct += test_correct.item()
        
        test_loss_avg = test_total_loss / test_step
        test_acc = 100 * (test_total_correct * 1.0 / test_total_data)
        
        print('avg_loss: {loss} acc: {acc}%'.format(loss=test_loss_avg, acc=test_acc))


def generate_args():
    parser = argparse.ArgumentParser(description='task3 test')

    parser.add_argument('--load_from', type=str, default='checkpoints/model_rnn_epoch_3_acc_65.09996155324875_2022-05-23-14:55:42', help='checkpoint path')
    parser.add_argument('--result', type=str, default='results/', help='path where result stored')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--data', type=str, default='../data/snli_1.0/', help='dataset path')
    parser.add_argument('--vector', type=str, default='glove.840B.300d.txt', help='vector file name')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = generate_args()
    test(args)