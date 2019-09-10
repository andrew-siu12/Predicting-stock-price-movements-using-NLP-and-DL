from base.base_model import BaseModel
import numpy as np
import tensorflow
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import Flatten, Dropout, Input, concatenate, BatchNormalization

class MLPModel(BaseModel):
    def __init__(self, config):
