import pandas as pd
import io
import re
import copy
import torch
from torchtext.vocab import build_vocab_from_iterator

import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


def df_from_lines(lines, categories, max_length=100):
    # load lines to DataFrame
    lines_list = [line.strip().split(' ::: ') for line in lines]
    df = pd.DataFrame(lines_list, columns=['0', 'title_year', 'genre', 'description'])
    df = df.drop(columns='0')
    # keep genre subset
    df = copy.deepcopy(df[df['genre'].isin(categories)]).reset_index(drop=True)

    # separate title / year
    df[['title', 'year']] = df['title_year'].str.rstrip(')').str.rsplit('(', expand=True, n=1)
    df['title'] = df['title'].str.strip()
    df['title'] = df['title'].str.strip('"')

    # clean descriptions
    df['clean_description'] = df['description'].apply(clean_text)
    df['clean_description_list'] = df['clean_description'].apply(text_to_list)
    # lemmatize descriptions using spacy
    df['lemmatized_description_list'] = df['clean_description'].apply(lemmatize_words)
    # truncate clean and lemmatized descriptions at max_length
    df['clean_description_list_trunc'] = df['clean_description_list'].apply(
        truncate_description_list, max_length=max_length)
    df['lemmatized_description_list_trunc'] = df['lemmatized_description_list'].apply(
        truncate_description_list, max_length=max_length)

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


def truncate_description_list(description_list, max_length=100):
    return description_list[:max_length]


def yield_tokens(data_iter): # for torchtext build_vocab_from_iterator
    for text in data_iter:
        yield text
        

def tokens_from_df(df, vocab):
    X_train = df['lemmatized_description_list_trunc'].apply(line_to_tokens, vocab=vocab)
    y_train = df['genre'].apply(category_to_index, genres=genres)
    return X_train, y_train,


def line_to_tokens(line, vocab):
    tokens = torch.tensor(vocab(line))
    return tokens


def category_to_index(category, genres): # target variable
    return torch.tensor([genres.index(category)], dtype=torch.long)


if __name__ == '__main__':

    genres = ['drama', 'documentary', 'comedy']#, 'horror', 'thriller', 'action']
    max_length=100
    min_freq=60

    # Train
    with io.open('data/imdb/train_data.txt', mode='r', encoding='utf-8') as f:
        train_lines = f.readlines()

    df = df_from_lines(train_lines, genres, max_length=max_length)
    
    vocab = build_vocab_from_iterator(
        yield_tokens(df['lemmatized_description_list_trunc']),
        min_freq=min_freq,
        specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    X_train, y_train = tokens_from_df(df, vocab)

    torch.save(vocab, 'saved_models/vocab{}.pt'.format(len(genres)))
    
    df.to_pickle('data/imdb_train_processed_{}_genres.pkl'.format(len(genres)))
    torch.save(X_train, 'data/X_train{}.pt'.format(len(genres)))
    torch.save(y_train, 'data/y_train{}.pt'.format(len(genres)))
    
    # Test
    with io.open('data/imdb/test_data_solution.txt', mode='r', encoding='utf-8') as f:
        test_lines = f.readlines()

    df_test = df_from_lines(test_lines, genres, max_length=max_length)

    X_test, y_test = tokens_from_df(df_test, vocab)
    
    df_test.to_pickle('data/imdb_test_processed_{}_genres.pkl'.format(len(genres)))    
    torch.save(X_test, 'data/X_test{}.pt'.format(len(genres)))
    torch.save(y_test, 'data/y_test{}.pt'.format(len(genres)))
