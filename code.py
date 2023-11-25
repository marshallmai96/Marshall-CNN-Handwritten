import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_train = np.expand_dims(x_train, axis = -1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
x_test = np.expand_dims(x_test, axis = -1)

inputs = tf.keras.layers.Input(shape=(28, 28, 1))
convolution1 = tf.keras.layers.Conv2D(2,(3,3), padding = 'valid', activation = tf.nn.relu)(inputs)
max_pool1 = tf.keras.layers.MaxPool2D((2,2), (2,2))(convolution1)
convolution2 = tf.keras.layers.Conv2D(4,(3,3), padding = 'valid', activation = tf.nn.relu)(max_pool1)
max_pool2 = tf.keras.layers.MaxPool2D((2,2), (2,2))(convolution2)
flatten = tf.keras.layers.Flatten()(max_pool2)
dense1 = tf.keras.layers.Dense(128, activation = tf.nn.relu)(flatten)
dense2 = tf.keras.layers.Dense(128, activation = tf.nn.relu)(dense1)
outputs = tf.keras.layers.Dense(10, activation = tf.nn.softmax)(dense2)

model = tf.keras.models.Model(inputs,outputs)
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train,y_train, epochs = 10)
