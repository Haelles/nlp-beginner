import pandas as pd
import torch
import torchtext
from torchtext import data
from torchtext.vocab import Vectors
import spacy


class SentimentDataset(data.Dataset):
    def __init__(self, file_path, text_field, label_field, is_test=False):
        data_df = pd.read_csv(file_path)
        print('read data from {path}'.format(path=file_path))
        print('data length:{length}'.format(length=len(data_df)))

        fields = [
                ('PhraseId', None),
                ('SentenceID', None), 
                ('Phrase', text_field),
                ('Sentiment', label_field)
            ]
        
        examples = []
        if not is_test:
            for cur_data in data_df.itertuples():
                examples.append(data.Example.fromlist([None, None, cur_data.Phrase, cur_data.Sentiment], fields))
        else:
            for cur_data in data_df.itertuples():
                examples.append(data.Example.fromlist([None, None, cur_data.Phrase, None], fields))
        
        super(SentimentDataset, self).__init__(examples, fields)
        



if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
    label_field = data.Field(sequential=False, use_vocab=False)
    # file_path = 'data/sentiment-analysis-on-movie-reviews/test.csv'
    # test_dataset = SentimentDataset(file_path, text_field, label_field, True)
    file_path = 'data/sentiment-analysis-on-movie-reviews/train.csv'
    train_dataset = SentimentDataset(file_path, text_field, label_field)

    # import pdb
    # pdb.set_trace()

    print("done")