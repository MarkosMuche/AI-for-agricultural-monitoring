from __future__ import absolute_import, division, print_function
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import tensorflowjs as tfjs
import keras
import pickle
import time
import tensorflow as tf
import cv2
import matplotlib as plt
import numpy as np
import os
#############################import image array neger
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
print(X.shape)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
X = X/255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',input_shape=(100,100,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, batch_size=12, epochs=10, validation_split=0.1)
model.save('mymodel.h5')
