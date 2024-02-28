import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras.backend as K #for some advanced functions   
from keras.optimizers import *
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
    features_array = np.load(x_path, mmap_mode='r+')
    
    # Load the labels list
    labels_array = np.load(y_path, mmap_mode='r+')
    
    
    return features_array, labels_array

X_train, Y_train = load_saved_dataset(x_train_save_path, y_train_padded_save_path)
X_dev, Y_dev = load_saved_dataset(x_dev_save_path, y_dev_padded_save_path)
X_test, Y_test = load_saved_dataset(x_test_save_path, y_test_padded_save_path)


characters = set(char for label in Y_dev for char in label)
num_classes = len(characters) + 1  # +1 for the CTC "blank" character
num_classes = 233  # xd idk
durations_path = '../dataset/fr/clip_durations.tsv'

clip_durations = pd.read_csv(durations_path, sep='\t', header=None)
# Display the first few entries to understand the data structure
clip_durations.head()
# Drop the first row which contains the header information

clip_durations_cleaned = clip_durations.drop(0)
# Convert the duration column to numeric, ignoring errors
clip_durations_cleaned[1] = pd.to_numeric(clip_durations_cleaned[1], errors='coerce')

# Find the maximum duration value
max_duration_ms = clip_durations_cleaned[1].max()
hop_length_ms = 10  # This is a typical value, but it should be set to whatever you used during feature extraction
max_pad_len = max_duration_ms / hop_length_ms
max_pad_len = np.ceil(max_pad_len).astype(int)

from keras.layers import Reshape

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(40, max_pad_len, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    # Reshape the output to be 3D (time steps, features) for the LSTM layer
    Reshape((-1, 19*32)),  # Assuming 19 is the number of time steps after pooling, and 32 is the number of filters
    LSTM(128, return_sequences=True, dropout=0.2),
    LSTM(128, dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])



model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['accuracy'])
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# You would convert your preprocessed data into a format suitable for model.fit, such as using tf.data.Dataset

# Evaluate the model
# Use the preprocessed data from dev.tsv and test.tsv to validate and test your model
model.save('speech-to-text-fr-model.keras')