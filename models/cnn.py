from base.base_model import BaseModel
import numpy as np
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPool1D
from tensorflow.keras.layers import Flatten, Dropout, Input, concatenate
from tensorflow.keras.models import Model
from utils.util import f1
import os

class CNNModel(BaseModel):
    def __init__(self, config, word_index):
        super(CNNModel, self).__init__(config)
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

        self.convs = []
        filter_sizes = [3, 4, 5]

        for filter_size in filter_sizes:
            conv_l = Conv1D(filters=config.filters, kernel_size=filter_size, activation="relu" )(self.word_emb)
            pool_l = MaxPool1D(pool_size=config.pool_size)(conv_l)
            self.convs.append(pool_l)
        self.merge_l = concatenate(self.convs, axis=1)
        self.merge_l = Dropout(config.Dropout)(self.merge_l)
        self.main = Flatten()(self.merge_l)

        self.aux_input = Input(shape=(3,), name="aux_input")
        self.concat = concatenate([self.main, self.aux_input])
        self.hidden_2 = Dense(config.Dense, activation="relu")(self.concat)
        self.hidden_2 = Dropout(config.Dropout_1)(self.hidden_2)
        self.hidden_3 = Dense(config.Dense_1, activation="relu")(self.hidden_2)
        self.hidden_3 = Dropout(config.Dropout_2)(self.hidden_3)
        self.output_layer = Dense(3, activation="sigmoid", name="output")(self.hidden_3)

        self.model = Model(inputs=[self.input, self.aux_input], outputs=[self.output_layer], name="cnn")
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1])