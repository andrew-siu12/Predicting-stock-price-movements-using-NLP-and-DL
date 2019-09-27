from base.base_model import BaseModel
import numpy as np
from tensorflow.keras.layers import Dense, Embedding, LSTM, Birdirectional
from tensorflow.keras.layers import Flatten, Dropout, Input, concatenate
from tensorflow.keras.models import Model
from keras_self_attention import SeqSelfAttention
from utils.util import f1
import os


class Att_BiLSTMModel(BaseModel):
    def __init__(self, config, word_index):
        super(Att_BiLSTMModel, self).__init__(config)
        self.max_sequence_length = config.max_sequence_length
        self.embedding_model_name = config.embedding_model_name
        self.embedding_dim = config.embedding_dim
        self.vocab_size = self.config.vocab_size
        self.word_index = word_index
        self.build_model(config)

    def get_embedding_matrix(self):
        embedding_index = {}

        with open(os.path.join("inputs/", self.embedding_model_name), 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype="float32")
                # key is string word, value is numpy array for vector
                embedding_index[word] = vector
        embedding_matrix = np.zeros((self.config.vocab_size, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def embedding_layer(self):
        embedding_matrix = self.get_embedding_matrix()

        return Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            weights=[embedding_matrix],
            input_length=self.max_sequence_length,
            trainable=False
        )

    def build_model(self, config):
        self.input = Input(shape=(self.max_sequence_length,), name="word_input")
        self.word_emb = self.embedding_layer()(self.input)

        self.lstm_l = Bidirectional(LSTM(units=config.units,
                                 dropout=config.lstm_Dropout,
                                 return_sequences=False))(word_emb)
        self.self_attention = SeqSelfAttention(attention_activation='sigmoid')(self.lstm_l)
        self.self_attention_flatten = Flatten()(self.self_attention)

        self.aux_input = Input(shape=(3,), name="aux_input")
        self.concat = concatenate([self.self_attention_flatten, self.aux_input])
        self.hidden_2 = Dense(config.Dense, activation="relu")(self.concat)
        self.hidden_2 = Dropout(config.Dropout_1)(self.hidden_2)
        self.hidden_3 = Dense(config.Dense_1, activation="relu")(self.hidden_2)
        self.hidden_3 = Dropout(config.Dropout_2)(self.hidden_3)
        self.output_layer = Dense(3, activation="sigmoid", name="output")(self.hidden_3)

        self.model = Model(inputs=[self.input, self.aux_input], outputs=[self.output_layer], name="cnn")
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1])