import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()
