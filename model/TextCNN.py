from torchtext import data
import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d
import spacy

import sys
sys.path.append(".")
from utils.iterators import build_iterator


class TextCNN(nn.Module):
    def __init__(self, text_field, label_type_num=5, embedded_dim=50, in_channels=1, out_channels=100, kernel_size=[2, 3, 4], stride=1, drop_rate=0.5):
        super(TextCNN, self).__init__()
        self.embedded_dim = embedded_dim
        self.word_embedding = nn.Embedding.from_pretrained(text_field.vocab.vectors)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=(k, embedded_dim)) for k in kernel_size])
        self.activate = nn.ReLU()
        self.drop_out = nn.Dropout(drop_rate)
        self.class_head = nn.Linear(out_channels * len(kernel_size), label_type_num)
    

    def forward(self, data):
        data_permute = data.Phrase.permute(1, 0)  # convert to [batch_size, length]
        data_vectors = self.word_embedding(data_permute).unsqueeze(1)  # [batch, 1, length, dim]
        
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
    
    file_path = 'data/sentiment-analysis-on-movie-reviews/train.csv'
    is_train = True    
    batch_size = 2

    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
    cur_iterator = build_iterator(file_path, text_field=text_field, batch_size=batch_size, device=device, is_train=is_train)

    model = TextCNN(text_field=text_field).to(device)

    batch = next(iter(cur_iterator))
    logits = model(batch)
    
    print('logits:{logits} done'.format(logits=logits.shape))