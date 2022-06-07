import argparse
import torch
import torchtext
from torchtext import data
import spacy
import numpy as np
import pandas as pd
import os
import time

from utils.iterators import build_iterator


def test(args):
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    nlp = spacy.load('en_core_web_sm')
    tokenize = lambda x: [tok.text for tok in nlp.tokenizer(x)]
    text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)

    batch_size = 8
    val_file_path = 'data/sentiment-analysis-on-movie-reviews/test.csv'
    test_iterator = build_iterator(val_file_path, text_field=text_field, batch_size=batch_size, device=device, train_tag='test')

    model = torch.load(args.load_from).to(device)
    model_name = args.load_from.split('/')[-1]
    if not os.path.exists(os.path.join(args.result, model_name)):
        os.makedirs(os.path.join(args.result, model_name))

    id_list = []
    sentiment_list = []
    with torch.no_grad():
        for idx, batch in enumerate(test_iterator):
            logits = model(batch)  # [batch, label_type_num]
            pred = torch.max(logits, dim=1)[-1]
            id_list.extend(batch.PhraseId.tolist())
            sentiment_list.extend(pred.tolist())

    res_dict = {'PhraseId': id_list, 'Sentiment': sentiment_list}
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(os.path.join(args.result, model_name, 'result.csv'), index=False)

    print('Test done. Time used {time}'.format(time=time.time() - start_time))


def generate_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--load_from', type=str, default='checkpoints/model_rnn_epoch_3_acc_65.09996155324875_2022-05-23-14:55:42', help='checkpoint path')
    parser.add_argument('--result', type=str, default='results/', help='path where result stored')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = generate_args()
    test(args)