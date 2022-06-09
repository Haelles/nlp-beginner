from torchtext import data
import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d
import spacy

import sys
sys.path.append(".")
from utils.iterators import build_iterator
from utils.utils import generate_mask
from layers import RNN_Dropout, BiLSTM


class ESIM(nn.Module):
    def __init__(self, text_field, label_type_num=5, embedded_dim=300, in_channels=1, out_channels=100, kernel_size=[2, 3, 4], stride=1, padding=2, dropout_rate=0.5):
        super(ESIM, self).__init__()
        self.embedded_dim = embedded_dim
        # TODO  待比较更新和不更新embedding的性能区别
        self.word_embedding = nn.Embedding.from_pretrained(text_field.vocab.vectors, padding_idx=1, freeze=False)
        # pytorch nn.Embedding文档提供了如下update padding vector的方法
        with torch.no_grad():
            self.word_embedding.weight[1, :] = torch.zeros(embedded_dim)

        if dropout_rate != 0:
            self.dropout_rate = dropout_rate
            self.dropout = RNN_Dropout(dropout_rate)

        self.encoder_lstm = BiLSTM()

        # padding should be a tuple
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=(k, embedded_dim), padding=(padding, 0)) for k in kernel_size])
        self.activate = nn.ReLU()
        self.class_head = nn.Linear(out_channels * len(kernel_size), label_type_num)
    

    def forward(self, premise_sequence, premise_length, hypothesis_sequence, hypothesis_length):
        # premise, hypothesis [b, l]
        # premise_length, hypothesis_length [b]
        # !在forward外面进行permute
        premise_mask = generate_mask(premise_sequence, premise_length)  # [b, l]
        hypothesis_mask = generate_mask(hypothesis_sequence, hypothesis_length)

        premise = self.word_embedding(premise_sequence)  # [b, l, d]
        hypothesis = self.word_embedding(hypothesis_sequence)

        if self.dropout_rate != 0:
            premise = self.dropout(premise)
            hypothesis = self.dropout(hypothesis)
        
        encoded_premise = self.encoder_lstm(premise, premise_length)  # should be [b, l, d]
        encoded_hypothesis = self.encoder_lstm(hypothesis, hypothesis_length)

        # attn:


        feature_vectors = []
        for module in self.convs:
            out_vectors = self.activate(module(data_vectors))  # out should be [batch, out_channels, length-kernel+1, 1]
            out_vectors = max_pool1d(out_vectors.squeeze(3), kernel_size=out_vectors.shape[2])  # out should be [batch, out_channels, 1]
            feature_vectors.append(out_vectors.squeeze(2))  # should be [batch, out_channels]

        cat_vectors = torch.cat(feature_vectors, 1)  # should be [batch, out_channels * len(kernel_size)]
        cat_vectors = self.drop_out(cat_vectors)
        logits = self.class_head(cat_vectors)  # should be  [batch, label_type_num]

        return logits


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # file_path = '../data/snli_1.0/snli_1.0_test.jsonl'
    file_path = '../data/snli_1.0/snli_1.0_dev.jsonl'
    batch_size = 32
    is_train = False
    
    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True)
    cur_iterator = build_iterator(file_path, text_field=text_field, batch_size=batch_size, device=device, is_train=is_train)

    model = ESIM(text_field=text_field).to(device)

    batch = next(iter(cur_iterator))
    logits = model(batch)
    
    print('logits:{logits} done'.format(logits=logits.shape))