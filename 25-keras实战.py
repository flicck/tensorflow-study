'''
cifar10 分类
    airplane
    automobile
    bird
    cat
    deer
    dog
    frog
    horse
    ship
    truck
 '''

import tensorflow as tf
from tensorflow.keras import  datasets,layers,optimizers,Sequential,metrics
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.cast(y, dtype=tf.int32)
    return x,y

batchsz = 128
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)
print('datasets:', x.shape, y.shape, x.min(), x.max())


