import numpy as np
import pandas as pd
import pickle
import io
import re
import copy
import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence 
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# define globals
GENRES = ['drama', 'documentary', 'comedy']#, 'horror', 'thriller', 'action']
MIN_FREQ= 60
MAX_LENGTH = 100


def df_from_lines(lines):
    # load lines to DataFrame
    lines_list = [line.strip().split(' ::: ') for line in lines]
    df = pd.DataFrame(lines_list, columns=['0', 'title_year', 'genre', 'description'])
    df = df.drop(columns='0')
    # keep genre subset
    df = copy.deepcopy(df[df['genre'].isin(GENRES)]).reset_index(drop=True)

    # separate title / year
    df[['title', 'year']] = df['title_year'].str.rstrip(')').str.rsplit('(', expand=True, n=1)
    df['title'] = df['title'].str.strip()
    df['title'] = df['title'].str.strip('"')

    # clean descriptions
    df['clean_description'] = df['description'].apply(clean_text)
    df['clean_description_list'] = df['clean_description'].apply(text_to_list)
    # lemmatize descriptions using spacy
    df['lemmatized_description_list'] = df['clean_description'].apply(lemmatize_words)
    # truncate clean and lemmatized descriptions at MAX_LENGTH
    df['clean_description_list_trunc'] = df['clean_description_list'].apply(
        truncate_description_list)
    df['lemmatized_description_list_trunc'] = df['lemmatized_description_list'].apply(
        truncate_description_list)

    return df


def clean_text(text):
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text


def text_to_list(text): # uses spacy nlp object
    return [str(token) for token in nlp(text)]


def lemmatize_words(text): # uses spacy nlp object
    return [token.lemma_ for token in nlp(text)]


def truncate_description_list(description_list):
    return description_list[:MAX_LENGTH]


def yield_tokens(data_iter): # for torchtext build_vocab_from_iterator
    for text in data_iter:
        yield text
        

def tokens_from_df(df, vocab):
    X_train = df['lemmatized_description_list_trunc'].apply(line_to_tokens, vocab=vocab)
    y_train = torch.tensor(df['genre'].apply(category_to_index))
    return X_train, y_train


def line_to_tokens(line, vocab):
    tokens = torch.tensor(vocab(line))
    return tokens


def category_to_index(category): # target variable
    return torch.tensor([GENRES.index(category)], dtype=torch.long)

## Train
with io.open('data/imdb/train_data.txt', mode='r', encoding='utf-8') as f:
    train_lines = f.readlines()
df = df_from_lines(train_lines)
# build vocab
vocab = build_vocab_from_iterator(
    yield_tokens(df['lemmatized_description_list_trunc']),
    min_freq=MIN_FREQ,
    specials=['<unk>','<pad>']
    )
vocab.set_default_index(vocab['<unk>'])
torch.save(vocab, 'saved_models/vocab_{}_{}_{}.pt'.format(len(GENRES), MIN_FREQ, MAX_LENGTH))
# tokenize descriptions
X_train, y_train = tokens_from_df(df, vocab)
# pad tokenized descriptions
X_train_pad = pad_sequence(list(X_train), batch_first=True, padding_value=vocab['<pad>'])
# get unpadded lengths
train_lengths = [len(x) for x in X_train]
# save
df.to_pickle('data/imdb_train_processed_{}.pkl'.format(len(GENRES)))
df[['genre', 'title']].to_pickle('data/imdb_train_processed_{}_app.pkl'.format(len(GENRES)))
torch.save(X_train_pad, 'data/X_train_{}_{}_{}.pt'.format(len(GENRES), MIN_FREQ, MAX_LENGTH))
torch.save(y_train, 'data/y_train_{}_{}_{}.pt'.format(len(GENRES), MIN_FREQ, MAX_LENGTH))
with open('data/train_lengths_{}_{}_{}.pkl'.format(len(GENRES), MIN_FREQ, MAX_LENGTH), 'wb') as f:
    pickle.dump(train_lengths, f)
    
print('Train set complete')

## Test
with io.open('data/imdb/test_data_solution.txt', mode='r', encoding='utf-8') as f:
    test_lines = f.readlines()
df_test = df_from_lines(test_lines)
# tokenize descriptions
X_test, y_test = tokens_from_df(df_test, vocab)
# pad tokenized descriptions
X_test_pad = pad_sequence(list(X_test), batch_first=True, padding_value=vocab['<pad>'])
# get unpadded lengths
test_lengths = [len(x) for x in X_test]
# save
df_test.to_pickle('data/imdb_test_processed_{}.pkl'.format(len(GENRES)))
df_test[['genre', 'title']].to_pickle('data/imdb_test_processed_{}_app.pkl'.format(len(GENRES)))
torch.save(X_test_pad, 'data/X_test_{}_{}_{}.pt'.format(len(GENRES), MIN_FREQ, MAX_LENGTH))
torch.save(y_test, 'data/y_test_{}_{}_{}.pt'.format(len(GENRES), MIN_FREQ, MAX_LENGTH))
with open('data/test_lengths_{}_{}_{}.pkl'.format(len(GENRES), MIN_FREQ, MAX_LENGTH), 'wb') as f:
    pickle.dump(test_lengths, f)
    
print('Test set complete')
