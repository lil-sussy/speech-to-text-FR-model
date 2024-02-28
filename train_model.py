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


from keras.utils import Sequence
import numpy as np
import pandas as pd
import os

class DataGenerator(Sequence):
    def __init__(self, x_set_path, y_set_path, batch_size=32, shuffle=True):
        # Load the entire dataset into memory if it fits, else modify as needed
        self.x_set = np.load(x_set_path, mmap_mode='r+')
        self.y_set = np.load(y_set_path, mmap_mode='r+')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x_set))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.x_set) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X = self.x_set[indexes]
        y = self.y_set[indexes]

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Parameters
batch_size = 32

# Parameters
params = {
    'dim': (40, 11279),
    'batch_size': 32,
    'n_channels': 1,
    'shuffle': True
}

# Datasets
partition = {
    'train': x_dev_save_path,  # Define your paths
    'validation': x_test_save_path
}# some dictionary containing your train/test split of file paths
labels = {
    'train': y_dev_padded_save_path,  # Define your paths
    'validation': y_test_padded_save_path
}# a dictionary containing all your labels for each file path

# Generators
training_generator = DataGenerator(partition['train'], labels['train'], batch_size)
validation_generator = DataGenerator(partition['validation'], labels['validation'], batch_size)

# Train model on dataset
history = model.fit(training_generator, validation_data=validation_generator, epochs=10)



# Define callbacks (optional)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.keras', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
]

# Start training
history = model.fit(
    training_generator, validation_data=validation_generator,
    epochs=10,  # You can adjust the number of epochs
    batch_size=32,  # And the batch size
    callbacks=callbacks,
    use_multiprocessing=True, workers=6,
    verbose=1
)

# After training, evaluate your model on the test set
test_loss, test_accuracy = model.evaluate(validation_generator, verbose=1)
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
