from torchtext import data
import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d
import spacy

import sys
sys.path.append(".")
from utils.iterators import build_iterator


class TextRNN(nn.Module):
    def __init__(self, text_field, label_type_num=5, embedded_dim=50, hidden_size=100, num_layers=1, bidirectional=True, drop_rate=0):
        super(TextRNN, self).__init__()
        self.embedded_dim = embedded_dim
        self.word_embedding = nn.Embedding.from_pretrained(text_field.vocab.vectors)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=embedded_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_rate, bidirectional=bidirectional)

        if bidirectional:
            self.class_head = nn.Linear(hidden_size * 2, label_type_num)
        else:
            self.class_head = nn.Linear(hidden_size, label_type_num)
    

    def forward(self, data):
        data_permute = data.Phrase.permute(1, 0)  # convert to [batch_size, length]
        data_vectors = self.word_embedding(data_permute)  # [batch, length, dim]
        
        if self.bidirectional:
            h0 = data_vectors.new_zeros((2 * self.num_layers, data_vectors.shape[0], self.hidden_size))
            c0 = data_vectors.new_zeros((2 * self.num_layers, data_vectors.shape[0], self.hidden_size))
        else:
            h0 = data_vectors.new_zeros((self.num_layers, data_vectors.shape[0], self.hidden_size))
            c0 = data_vectors.new_zeros((self.num_layers, data_vectors.shape[0], self.hidden_size))

        feature_vectors, _ = self.lstm(data_vectors, (h0, c0))  # should be [batch, length, hidden_size]
        logits = self.class_head(feature_vectors[:, -1, :])  # should be  [batch, label_type_num]

        return logits


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    file_path = 'data/sentiment-analysis-on-movie-reviews/train.csv'
    is_train = True    
    batch_size = 8

    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
    cur_iterator = build_iterator(file_path, text_field=text_field, batch_size=batch_size, device=device, train_tag='train')

    model = TextRNN(text_field=text_field).to(device)

    batch = next(iter(cur_iterator))
    logits = model(batch)
    
    print('logits:{logits} done'.format(logits=logits.shape))