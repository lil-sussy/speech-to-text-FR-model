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

# Assuming the model is defined as 'model' and compiled as per your earlier message
model = load_model('speech-to-text-fr-model.keras')


def load_saved_dataset(x_path, y_path):
    # Load the features array
    features_array = np.load(x_path)
    
    # Load the labels list
    labels_array = np.load(y_path)
    
    return features_array, labels_array

X_train, Y_train = load_saved_dataset(x_train_save_path, y_train_padded_save_path)
X_dev, Y_dev = load_saved_dataset(x_dev_save_path, y_dev_padded_save_path)
X_test, Y_test = load_saved_dataset(x_test_save_path, y_test_padded_save_path)


# Define callbacks (optional)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.keras', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
]

# Start training
history = model.fit(
    X_dev, Y_dev,
    validation_data=(X_test, Y_test),
    epochs=10,  # You can adjust the number of epochs
    batch_size=32,  # And the batch size
    callbacks=callbacks,
    verbose=1
)

# After training, evaluate your model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Optionally, you can plot the training history:
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
