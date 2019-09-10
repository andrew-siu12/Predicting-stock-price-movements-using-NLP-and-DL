import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataGenerator:
    def __init__(self, config):
        padding = config.padding
        max_sequence_length = config.max_sequence_length
        data_file = "../inputs/preprocessed_news.csv"

        data_df = pd.read_csv("data_file")
        cols = ["price_change_short", "price_change_mid", "price_change_long"]
        price_feat_data = data_df[cols]
        labels = data_df['signal']
        labels = pd.get_dummies(columns=['signal'], data=labels)

        docs = data_df["combined_tokens"].apply(literal_eval)

        t = Tokenizer()
        t.fit_on_texts(docs)
        sequences = t.texts_to_sequences(docs)
        docs = pad_sequences(sequences=sequences,
                             maxlen=max_sequence_length, padding=padding)

        self.word_index = t.word_index
        config.vocab_size = len(self.word_index)

        self.price_train, self.price_test, self.label_train, \
            self.label_test, self.docs_train, self.docs_test = train_test_split(
                price_feat_data, labels, docs, stratify=labels,
                test_size=config.test_size, random_state=42)

    def get_train_data(self):
        return self.price_train, self.label_train, self.docs_train

    def get_test_data(self):
        return self.price_test, self.label_test, self.docs_test

    def get_word_index(self):
        return self.word_index

