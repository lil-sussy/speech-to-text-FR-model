import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras.backend as K #for some advanced functions   
from keras.optimizers import *

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
# train_df = pd.read_csv('path/to/train.tsv', sep='\t')
# dev_df = pd.read_csv('path/to/dev.tsv', sep='\t')
# test_df = pd.read_csv('path/to/test.tsv', sep='\t')

audio_files_path = '../dataset/fr/clips/'
train_tsv_path = '../dataset/fr/train.tsv'
dev_tsv_path = '../dataset/fr/dev.tsv'
test_tsv_path = '../dataset/fr/test.tsv'
import os
# Function to extract MFCC features
durations_path = '../dataset/fr/clip_durations.tsv'

clip_durations = pd.read_csv(durations_path, sep='\t', header=None)
# Display the first few entries to understand the data structure
clip_durations.head()
# Drop the first row which contains the header information

clip_durations_cleaned = clip_durations.drop(0)
# Convert the duration column to numeric, ignoring errors
clip_durations_cleaned[1] = pd.to_numeric(clip_durations_cleaned[1], errors='coerce')
max_duration_ms = clip_durations_cleaned[1].max()
hop_length_ms = 10  # This is a typical value, but it should be set to whatever you used during feature extraction
max_pad_len = max_duration_ms / hop_length_ms
max_pad_len = np.ceil(max_pad_len).astype(int)

def extract_features(audio_file, max_length, n_mfcc=40):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(audio_file, sr=None)
        # Extract MFCC features with 40 features instead of 13
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # Transpose the matrix to get time steps as the first dimension
        mfccs = mfccs.T
        # Pad or truncate as necessary
        if mfccs.shape[0] < max_length:
            pad_width = max_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant')
        elif mfccs.shape[0] > max_length:
            mfccs = mfccs[:max_length, :]
    except Exception as e:
        print("Error encountered while parsing file: ", audio_file)
        return None 
    # Expand the dimensions to include a channel dimension
    mfccs = np.expand_dims(mfccs, -1)
    return mfccs

import numpy as np
import pickle
import os
from tqdm import tqdm

def generate_dataset(tsv_path, audio_files_path, x_save_path, y_save_path, max_pad_len):
    df = pd.read_csv(tsv_path, sep='\t')
    labels = []
    
    # Initialize the memmap array
    features_shape = (len(df), max_pad_len, 40, 1)
    features_array = np.lib.format.open_memmap(x_save_path, mode='w+', dtype=np.float32, shape=features_shape)
    
    pbar = tqdm(total=len(df), desc='Processing', unit='files')
    
    for index, row in df.iterrows():
        file_path = os.path.join(audio_files_path, row['path'])
        mfccs = extract_features(file_path, max_pad_len)
        
        if mfccs is not None:
            # Assign the mfccs to the memmap array at the correct index
            features_array[index] = mfccs
            labels.append(row['sentence'])
        pbar.update(1)
    pbar.close()
    
    # Save the labels array
    labels_array = np.array(labels)
    with open(y_save_path, 'wb') as f:
        pickle.dump(labels_array, f)
    return features_array, labels_array

# Define paths
x_train_save_path = 'data/X_train.npy'
y_train_save_path = 'data/Y_train.pickle'
x_dev_save_path = 'data/X_dev.npy'
y_dev_save_path = 'data/Y_dev.pickle'
x_test_save_path = 'data/X_test.npy'
y_test_save_path = 'data/Y_test.pickle'

print("Loading dev dataset...")
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
# Load the datasets
# print("Loading train dataset...")
# X_train, Y_train = load_dataset(train_tsv_path, audio_files_path, x_train_save_path, y_train_save_path, max_pad_len)
X_dev, Y_dev = generate_dataset(dev_tsv_path, audio_files_path, x_dev_save_path, y_dev_save_path, max_pad_len)
print("Loading test dataset...")
X_test, Y_test = generate_dataset(test_tsv_path, audio_files_path, x_test_save_path, y_test_save_path, max_pad_len)

