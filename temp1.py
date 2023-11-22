import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

#plt.imshow(x_train[0], cmap='Greys')

# Flatten the input data
x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))

model_lr = tf.keras.models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model_lr.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

y_onehot_train = tf.one_hot(y_train, 10)
model_lr.fit(x_train, y_onehot_train, epochs=2)
model_lr.summary()
