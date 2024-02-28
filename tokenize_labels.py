import tensorflow as tf
from tensorflow.python.keras.models import *
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras.backend as K #for some advanced functions   
from keras.optimizers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

audio_files_path = '../dataset/fr/clips/'
train_tsv_path = '../dataset/fr/train.tsv'
dev_tsv_path = '../dataset/fr/dev.tsv'
test_tsv_path = '../dataset/fr/test.tsv'

x_train_save_path = 'data/X_train.npy'
y_train_save_path = 'data/Y_train.pickle'
y_train_padded_save_path = 'data/Y_train_padded.npy'
x_dev_save_path = 'data/X_dev.npy'
y_dev_save_path = 'data/Y_dev.pickle'
y_dev_padded_save_path = 'data/Y_dev_padded.npy'
x_test_save_path = 'data/X_test.npy'
y_test_save_path = 'data/Y_test.pickle'
y_test_padded_save_path = 'data/Y_test_padded.npy'


def load_saved_dataset(x_path, y_path):
    # Load the features array
    features_array = np.load(x_path)
    
    # Load the labels list
    with open(y_path, 'rb') as f:
        labels_array = pickle.load(f)
    
    return features_array, labels_array

X_train, Y_train = load_saved_dataset(x_train_save_path, y_train_save_path)
X_dev, Y_dev = load_saved_dataset(x_dev_save_path, y_dev_save_path)
X_test, Y_test = load_saved_dataset(x_test_save_path, y_test_save_path)


# Initialize the tokenizer with char-level encoding
tokenizer = Tokenizer(char_level=True)

# Fit the tokenizer on the sentences
tokenizer.fit_on_texts(Y_train)

# Convert sentences to sequences of integers
Y_train_seq = tokenizer.texts_to_sequences(Y_train)
Y_dev_seq = tokenizer.texts_to_sequences(Y_dev)
Y_test_seq = tokenizer.texts_to_sequences(Y_test)

# Pad the sequences to have the same length
Y_train_padded = pad_sequences(Y_train_seq, padding='post')
Y_dev_padded = pad_sequences(Y_dev_seq, padding='post', maxlen=Y_train_padded.shape[1])
Y_test_padded = pad_sequences(Y_test_seq, padding='post', maxlen=Y_train_padded.shape[1])
np.save(y_train_padded_save_path, Y_train_padded)
np.save(y_dev_padded_save_path, Y_dev_padded)
np.save(y_test_padded_save_path, Y_test_padded)