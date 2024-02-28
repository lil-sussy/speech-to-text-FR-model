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
x_dev_save_path = 'data/X_dev.npy'
y_dev_save_path = 'data/Y_dev.pickle'
x_test_save_path = 'data/X_test.npy'
y_test_save_path = 'data/Y_test.pickle'

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


characters = set(char for label in Y_dev for char in label)
num_classes = len(characters) + 1  # +1 for the CTC "blank" character
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

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(40, max_pad_len, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    # Remove Flatten() here to maintain the temporal dimension
    TimeDistributed(Flatten()),  # Optionally use TimeDistributed wrapper if needed
    LSTM(128, return_sequences=True, dropout=0.2),
    LSTM(128, dropout=0.2),
    Flatten(),  # You can add Flatten() here if you're moving to Dense layers next
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# You would convert your preprocessed data into a format suitable for model.fit, such as using tf.data.Dataset

# Evaluate the model
# Use the preprocessed data from dev.tsv and test.tsv to validate and test your model
model.save('speech-to-text-fr-model.keras')