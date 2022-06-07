import pandas as pd
import torch
import torchtext
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import jsonlines


class SnliDataset(data.Dataset):
    def __init__(self, file_path, text_field, label_field):
        fields = {
                'sentence1': ('premise', text_field),  # data.Example.fromdict里面涉及name, field = val，需要赋一个名字
                'sentence2': ('hypothesis',text_field), 
                'gold_label': ('label', label_field)
            }

        examples = []
        with open(file_path, 'r+', encoding='utf-8') as f:
            cnt = 0
            total_cnt = 0
            print('read data from {path}'.format(path=file_path))
            for item in jsonlines.Reader(f):
                total_cnt += 1
                if item['gold_label'] == '-':
                    print(item['pairID'])
                    print(item['sentence1'])
                    continue
                examples.append(data.Example.fromdict(item, fields))
                cnt += 1
            print('data length:{length}'.format(length=cnt))
            print('total file length:{length}'.format(length=total_cnt))
        
        super(SnliDataset, self).__init__(examples=examples, fields=fields)
        

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
    label_field = data.Field(sequential=False, use_vocab=False)
    # file_path = '../data/snli_1.0/snli_1.0_test.jsonl'
    # test_dataset = SnliDataset(file_path, text_field, label_field)
    file_path = '../data/snli_1.0/snli_1.0_dev.jsonl'
    dev_dataset = SnliDataset(file_path, text_field, label_field)
    # file_path = 'data/snli_1.0/snli_1.0_train.jsonl'
    # train_dataset = SnliDataset(file_path, text_field, label_field)

    # import pdb
    # pdb.set_trace()

    print("done")