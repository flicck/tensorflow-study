import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
'''
keras
    Sequential 容器
    layers.layer 层容器
    Model 模型父类 Sequential的父类也是Model
'''
# 自己实现一个Dense层，注意默认的是layers.Dense，自定义的话可以在层里操作
class MyDense(layers.Layer):
    def __init__(self,inp_dim, outp_dim):
        super(MyDense,self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    # training参数可用于区分是训练还是测试
    def call(self, inputs, training = None):
        out = input@self.kernel + self.bias
        return out

# 默认是Sequential，自定义的话可以在层与层之间进行操作
class MyModel(keras.Model):
    def __init__(self):
        # 调用父类构造器
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 18)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x
