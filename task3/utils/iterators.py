from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
import torch
import torch.nn as nn
import spacy

import sys
sys.path.append(".")
from dataset.snli import SnliDataset


def build_iterator(file_path, batch_size=1, device=-1, is_train=True, text_field=None, spcay_load='en_core_web_sm', vector_file_name='glove.840B.300d.txt'):
    if text_field is None:
        nlp = spacy.load(spcay_load)
        # TODO 后续可以看一下去掉标点符号的效果
        tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
        text_field = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True)  # 为了后续构造mask
    label_field = data.Field(sequential=False, use_vocab=False, is_target=False)  # 用data.LabelField应该也可以

    cur_dataset = SnliDataset(file_path, text_field, label_field)

    vectors = Vectors(name=vector_file_name, cache='../.vector_cache/')
    # vectors.unk_init = nn.init.xavier_uniform_  # 原文中说用的Gaussian samples
    vectors.unk_init = nn.init.normal_

    text_field.build_vocab(cur_dataset, vectors=vectors)
    # import pdb
    # pdb.set_trace()
    if is_train:
        cur_iterator = BucketIterator(dataset=cur_dataset, batch_size=batch_size, device=device, sort_key=lambda x: len(x.premise) + len(x.hypothesis), shuffle=True, sort_within_batch=False, repeat=False)
    else:
        cur_iterator = Iterator(dataset=cur_dataset, batch_size=batch_size, device=device, train=False, sort=False, repeat=False)
    return cur_iterator


if __name__ == '__main__':
    # file_path = '../data/snli_1.0/snli_1.0_test.jsonl'
    file_path = '../data/snli_1.0/snli_1.0_dev.jsonl'
    batch_size = 32
    is_train = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cur_iterator = build_iterator(file_path, batch_size=batch_size, device=device, is_train=is_train)
    import pdb
    pdb.set_trace()
    # dir(batch): 'batch_size', 'dataset', 'fields', 'fromvars', 'hypothesis', 'input_fields', 'label', 'premise', 'target_fields' etc.
    # 注意下面一列是一个句子！！要取[:, 0]
    # batch.hypothesis[0].shape torch.Size([16, 32]) torch.Size([20, 32])
    # p batch.premise[0].shape torch.Size([28, 32]) torch.Size([34, 32])
    # p batch.premise[1].shape torch.Size([32])
    batch = next(iter(cur_iterator))
    
