import pandas as pd
from sklearn.model_selection import train_test_split


def generate_csv(data_path, train_path, test_path, val_size=0.2):
    train=pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')
    
    origin_data = train[['PhraseId', 'SentenceId', 'Phrase']]
    origin_label = train[['Sentiment']]
    train_data, val_data, train_label, val_label = train_test_split(
                                            origin_data, origin_label, val_size=0.2, random_state=0)
    
    train_final = pd.merge(train_data, train_label, left_index=True, right_index=True)
    val_final = pd.merge(val_data, val_label, left_index=True, right_index=True)
    print('train_final: {len}'.format(len=len(train_final)))
    print('val_final: {len}'.format(len=len(val_final)))
    print('test_final: {len}'.format(len=len(test)))
    
    train_final.to_csv(data_path + 'train.csv', index=False)
    val_final.to_csv(data_path + 'val.csv', index=False)
    test.to_csv(data_path + 'test.csv', index=False) 


if __name__ == '__main__':
    data_path = 'data/sentiment-analysis-on-movie-reviews/'
    train_path = 'data/sentiment-analysis-on-movie-reviews/train.tsv'
    test_path = 'data/sentiment-analysis-on-movie-reviews/test.tsv'
    generate_csv(data_path, train_path, test_path)
