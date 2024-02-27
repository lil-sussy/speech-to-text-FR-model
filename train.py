import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load the metadata
# Example function to extract features (e.g., MFCC)
def extract_features3(audio_path):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def extract_features2(file_path, max_pad_len=400):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None 
    return mfccs


# Example function to encode transcription
# def encode_transcription(text):
#     # Create a character to index mapping and encode
#     char_to_index = {' ': 0, 'a': 1, 'b': 2, ...}  # Continue for all characters
#     return [char_to_index[char] for char in text]

# Preprocess and prepare the dataset for training
# This would involve applying the extract_features and encode_transcription functions to your audio and text data

# Define your model architecture here using tf.keras...

# Compile the model

# Assuming `audio_paths` is a list of file paths to your audio clips
# mfcc_lengths = [librosa.feature.mfcc(y=librosa.load(p)[0]).shape[1] for p in audio_paths]
# max_pad_len = max(mfcc_lengths)  # This is the maximum length of MFCC features in your dataset

# unique_tokens = set(''.join(train_df['transcription']))  # Assuming 'transcription' column contains all transcriptions
# num_classes = len(unique_tokens) + 1  # +1 for the blank label in CTC

# Replace with the actual path to your audio files and TSV files
train_df = pd.read_csv('path/to/train.tsv', sep='\t')
dev_df = pd.read_csv('path/to/dev.tsv', sep='\t')
test_df = pd.read_csv('path/to/test.tsv', sep='\t')

audio_files_path = '../fr/clips/'
train_tsv_path = './dataset/train.tsv'
dev_tsv_path = './dataset/dev.tsv'
test_tsv_path = './dataset/test.tsv'
import os
# Function to extract MFCC features
def extract_features(audio_file, n_mfcc=13):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(audio_file, sr=None)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # Mean normalization to balance the dataset
        mfccs = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", audio_file)
        return None 
    return mfccs

# Function to load a dataset from a TSV file
def load_dataset(tsv_path):
    # Read the TSV file
    df = pd.read_csv(tsv_path, sep='\t')
    # Initialize empty lists for features and labels
    features = []
    labels = []
    
    # Loop over each row in the DataFrame
    for _, row in df.iterrows():
        # Get the file path for the audio file (assuming 'path' is the column with file paths)
        file_path = os.path.join(audio_files_path, row['path'])
        # Extract features
        mfccs = extract_features(file_path)
        
        if mfccs is not None:
            # Append the features and label to the respective lists
            features.append(mfccs)
            labels.append(row['sentence'])  # Assuming 'sentence' is the column with transcriptions
    
    return np.array(features), labels

# Load the datasets
print("Loading train dataset...")
X_train, y_train = load_dataset(train_tsv_path)
print("Loading dev dataset...")
X_dev, y_dev = load_dataset(dev_tsv_path)
print("Loading test dataset...")
X_test, y_test = load_dataset(test_tsv_path)

characters = set(char for label in y_train for char in label)
num_classes = len(characters) + 1  # +1 for the CTC "blank" character
durations_path = './dataset/clip_durations.tsv'

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

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(40, max_pad_len, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    LSTM(128, return_sequences=True, dropout=0.2),
    LSTM(128, dropout=0.2),
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
