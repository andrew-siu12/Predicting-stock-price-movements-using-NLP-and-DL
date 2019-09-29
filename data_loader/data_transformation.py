import os
import datetime
import numpy as np
import re
import warnings
import pandas as pd
import calendar
import pickle
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import spacy


def load_data(data_path):
    """
    Load stock_price and news into pandas dataframe
    :param data_path:
    :return:  stock_news_df, stock_price_df: pandas Dataframe.
    """
    stock_news_df = pd.read_csv(data_path + 'news_reuters.csv', header=None,
                                names=['tickers', 'company', 'date', 'headline', 'first_sent', 'priority'])

    with open(data_path + 'stockReturns.json') as f:
        stock_price = json.load(f)
    stock_price_df = pd.DataFrame(stock_price)

    return stock_news_df, stock_price_df


def transform_stock_price(price_df, duration):
    """
    Return dataframe with  columns price_change_duration,  tickers names and date. If duration is short: return extra column signal which is the
    labels for our classification task
    :param price_df:
    :param duration:  str. either short, mid or long
    :return:  transform_df:
    """
    transform_df = price_df[duration].apply(pd.Series)
    transform_df = transform_df.stack().rename('price_change' + '_' + duration).reset_index()
    transform_df.rename(columns={'level_0': 'tickers', 'level_1': 'date'}, inplace=True)
    transform_df.date = transform_df.date.astype('int64')

    if duration == 'short':
        transform_df['signal'] = transform_df['price_change' + '_' + duration] \
            .map(lambda x: "stay" if -1 < x < 1 else ("up" if x > 1 else "down"))
    return transform_df


def combine_stock_news(news_df, price_df):
    """
        Combine price and news_df with extra columns  'price_change_short', 'price_change_mid', 'signal',
       'price_change_long'
    :param news_df: pandas datafrmae
    :param price_df: pandas dataframe
    :return: combined_df: pandas dataframe. Merged of news and stock price df.
    """
    combined_df = news_df.copy()

    durations = price_df.columns
    for duration in durations:
        price_duration_df = transform_stock_price(price_df, duration)
        combined_df = pd.merge(left=combined_df, right=price_duration_df,
                               on=['date', 'tickers'], how='inner')
    return combined_df


def to_csv(data_path):
    """
    Save  the combined df into csv file
    :param data_path:
    :return:
    """
    news_df, price_df = load_data(data_path)

    combined_df = combine_stock_news(news_df, price_df)

    combined_df.to_csv(data_path + "news_price_df.csv")


def cleanup_text(sent):
    """
    Return a cleaned list of  text
    :param sent:  a  list of strings
    :return:
    """
    monthStrings = list(calendar.month_name)[1:] + list(calendar.month_abbr)[1:]
    monthPattern = '|'.join(monthStrings)
    sent = re.sub(r'\s+', ' ', str(sent)).strip()
    sent = re.sub(r'\/+', '', sent)
    sent = re.sub(r'U.S.', 'United States', sent)
    sent = re.sub(r'CORRECTED-', '', sent)
    sent = re.sub(r'^(\W?[A-Z\s\d]+\b-?)', '', sent)
    sent = re.sub(r'^ ?\W ', '', sent)
    sent = re.sub(r'(\s*-+\s*[A-Za-z]+)$', '', sent)
    sent = re.sub(r"(\'+[A-Z1-9]+\'*)$", '', sent)
    sent = re.sub(r"[$'|]+", '', sent)
    sent = re.sub(r'({}) \d+'.format(monthPattern), '', sent)

    sent = sent.lower().strip()

    return sent


def spacy_tokenize(df, col):
    """
    Remove stop words, lemmatize the cleaned text and tokenize
    :param df:
    :param col:  either "headline" or "first_sent" col
    :return:  filtered_tokens
    """
    nlp = English()
    STOP_WORDS = construct_stop_words()
    sentences = df[col].tolist()
    docs = []
    for sent in sentences:
        docs.append(cleanup_text(sent))

    def token_filter(token, stop_words):
        return not (token.is_punct or token.is_stop or token.is_space)

    try:
        filtered_tokens = []
        for doc in nlp.pipe(docs):
            tokens = [tok.lemma_ for tok in doc if token_filter(tok, STOP_WORDS)]
            tokens = [tok for tok in tokens if not re.search('[\$1-9]+', tok)]
            filtered_tokens.append(tokens)

        return filtered_tokens
    except Exception as e:
        raise e


def construct_stop_words():
    """
    Update the spacy stopwords list
    :return:
    """
    stop_words_list = ["uk", "ceo", "apple", "wal", "st", "q1", "q2", "q3", "q4",
                       "bp", "wednesday", "tuesday", "monday", "thursday", "friday", "sept", "johnson", "inc",
                       "david", "amazon.com"]

    for words in stop_words_list:
        STOP_WORDS.add(words)

    return STOP_WORDS

def limit_to_one_news(news):
    """
    Return one news per date for each ticker
    :param news:
    :return:
    """
    if news.shape[0] > 1:
        if 'topStory' in news['priority'].unique():
            news = news.loc[news['headline'] == "topStory"]
        if news.shape[0] > 1:
            news = news.sample(n=1, random_state=12)
    return news


if __name__ == "__main__":
    data_path = "../inputs/"
    news_df, price_df = load_data(data_path)
    stock_news_df = combine_stock_news(news_df, price_df)
    print("Succesfully combined news and stocks df....")

    filtered_tokens = spacy_tokenize(stock_news_df, 'headline')
    filtered_tokens_first_sent = spacy_tokenize(stock_news_df, "first_sent")
    transformed_news_df = stock_news_df.copy()
    transformed_news_df['headlines_tokens'] = filtered_tokens
    transformed_news_df['first_sent_tokens'] = filtered_tokens_first_sent
    transformed_news_df['head_tok_len'] = transformed_news_df['headlines_tokens'].map(lambda x: len(x))
    transformed_news_df['sent_tok_len'] = transformed_news_df['first_sent_tokens'].map(lambda x: len(x))

    transformed_news_df = transformed_news_df.groupby(['tickers', 'date']).apply(limit_to_one_news)
    transformed_news_df['combined_tokens'] = transformed_news_df.headlines_tokens \
                                             + transformed_news_df.first_sent_tokens
    transformed_news_df['combined_len'] = transformed_news_df.head_tok_len \
                                          + transformed_news_df.sent_tok_len
    transformed_news_df = transformed_news_df.droplevel([0, 1])

    data_types_df = transformed_news_df.dtypes.astype(str).to_dict()
    with open('inputs/data_type_key.json', 'w', encoding='utf-8') as f:
        json.dump(data_types_df, f)

    transformed_news_df.to_csv("inputs/preprocessed_news.csv")
    print("Finished saving the transformed news df...")