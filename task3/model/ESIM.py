from torchtext import data
import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d
import spacy

import sys
sys.path.append(".")
from utils.iterators import build_iterator
from utils.utils import generate_mask, masked_max_pooling
from layers import RNN_Dropout, BiLSTM, Attn


class ESIM(nn.Module):
    def __init__(self, text_field, num_class=3, embedded_dim=300, hidden_size=300, dropout_rate=0.5):
        super(ESIM, self).__init__()
        self.embedded_dim = embedded_dim
        self.num_class = num_class
        self.hidden_size = hidden_size
        # TODO  待比较更新和不更新embedding的性能区别
        self.word_embedding = nn.Embedding.from_pretrained(text_field.vocab.vectors, padding_idx=1, freeze=False)
        # pytorch nn.Embedding文档提供了如下update padding vector的方法
        with torch.no_grad():
            self.word_embedding.weight[1, :] = torch.zeros(embedded_dim)

        if dropout_rate != 0.0:
            self.dropout_rate = dropout_rate
            self.dropout = RNN_Dropout(dropout_rate)
        else:
            self.dropout_rate = 0.0

        self.encoder_lstm = BiLSTM()
        
        self.attn = Attn()
        # TODO 有没有必要在LSTM输出的2*hidden_size处进行融合呢
        self.mapping = nn.Sequential(
            nn.Linear(2*4*hidden_size, hidden_size),
            nn.ReLU()
        )

        self.composition = BiLSTM()

        self.classification = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(8*hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(hidden_size, num_class)
        )


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
        
        encoded_premise = self.encoder_lstm(premise, premise_length)  # should be [b, l, 2*hidden_size]
        encoded_hypothesis = self.encoder_lstm(hypothesis, hypothesis_length)

        # attn: output should be [b, l, 2*hidden_size]
        # 实际上分别是[b, l1/l2, 2*hidden_size]
        attn_premise, attn_hypothesis = self.attn(encoded_premise, premise_mask, encoded_hypothesis, hypothesis_mask)

        # enhancement of local inference information
        # should be [b, l, 2*hidden_size*4]
        enhanced_premise = torch.cat([encoded_premise, attn_premise, encoded_premise - attn_premise, encoded_premise * attn_premise], dim=-1)
        enhanced_hypothesis = torch.cat([encoded_hypothesis, attn_hypothesis, encoded_hypothesis - attn_hypothesis, encoded_hypothesis * attn_hypothesis], dim=-1)

        mapped_premise = self.mapping(enhanced_premise)  # should be [b, l, hidden_size]
        mapped_hypothesis = self.mapping(enhanced_hypothesis)

        if self.dropout_rate != 0:
            mapped_premise = self.dropout(mapped_premise)
            mapped_hypothesis = self.dropout(mapped_hypothesis)

        v_a = self.composition(mapped_premise, premise_length)  # should be [b, l, 2*hidden_size]
        v_b = self.composition(mapped_hypothesis, hypothesis_length)

        # avg_pooling, max_pooling
        # v_a_ave/v_a_max etc. should be [b, 2*hidden_size]
        # [b, 2*hidden_size] / [b, 1]
        # v_a_ave = v_a.sum(dim=1) / premise_length.unsqueeze(1)  # TODO 这里用非浮点数的premise_length tensor会影响结果吗
        # v_b_ave = v_b.sum(dim=1) / hypothesis_length.unsqueeze(1)
        
        # 使用以防万一的写法，即使padding不为0也可获得准确ave结果
        v_a_ave = torch.sum(v_a * (premise_mask.unsqueeze(1).transpose(2, 1)), dim=1) / torch.sum(premise_mask, dim=1, keepdim=True)
        v_b_ave = torch.sum(v_b * (hypothesis_mask.unsqueeze(1).transpose(2, 1)), dim=1) / torch.sum(hypothesis_mask, dim=1, keepdim=True)

        v_a_max, _ = masked_max_pooling(v_a, premise_mask, -1e7).max(dim=1)  # should be [b, 2*hidden_size]
        v_b_max, _ = masked_max_pooling(v_b, hypothesis_mask, -1e7).max(dim=1)

        final_v = torch.cat([v_a_ave, v_a_max, v_b_ave, v_b_max], dim=-1)  # should be [b, 8*hidden_size]

        # classification
        logits = self.classification(final_v)  # should be [b, num_class]

        # TODO 存疑
        prediction = nn.functional.softmax(logits, dim=1)

        return logits, prediction


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
    premise, premise_length = batch.premise
    premise = premise.transpose(0, 1).contiguous()
    hypothesis, hypothesis_length  = batch.hypothesis
    hypothesis = hypothesis.transpose(0, 1).contiguous()  # should be [b, l]
    
    label = batch.label  # [batch] e.g. torch.Size([32])
    # import pdb
    # pdb.set_trace()
    logits, prediction = model(premise, premise_length, hypothesis, hypothesis_length)
    
    print('logits:{logits} done'.format(logits=logits.shape))