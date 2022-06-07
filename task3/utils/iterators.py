from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import spacy

import sys
sys.path.append(".")
from dataset.snli import SnliDataset


def build_iterator(file_path, text_field=None, batch_size=1, shuffle=True, spcay_load='en_core_web_sm', vector_file_name='glove.840B.300d.txt', train_tag='train', device=-1):
    if text_field is None:
        nlp = spacy.load(spcay_load)
        tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
        text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
    label_field = data.Field(sequential=False, use_vocab=False)

    cur_dataset = SentimentDataset(file_path, text_field, label_field, train_tag)

    vectors = Vectors(name=vector_file_name)
    vectors.unk_init = nn.init.xavier_uniform_

    text_field.build_vocab(cur_dataset, vectors=vectors)
    if train_tag == 'train':
        cur_iterator = BucketIterator(dataset=cur_dataset, batch_size=batch_size, device=device, sort_key=lambda x: len(x.Phrase), shuffle=True, sort_within_batch=False, repeat=False)
    else:
        cur_iterator = Iterator(dataset=cur_dataset, batch_size=batch_size, device=device, train=False, sort=False, repeat=False)
    return cur_iterator


if __name__ == '__main__':
    file_path = 'data/sentiment-analysis-on-movie-reviews/test.csv'
    batch_size = 2
    is_train = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cur_iterator = build_iterator(file_path, batch_size=batch_size, device=device, is_train=is_train)
    # import pdb
    # pdb.set_trace()
    batch = next(iter(cur_iterator))
    
