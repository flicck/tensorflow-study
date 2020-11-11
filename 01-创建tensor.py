'''
from numpy list
zeros ones
fill
random
constant
Application
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras

tf.convert_to_tensor(np.ones([2, 3]))
tf.convert_to_tensor([1, 2])
tf.convert_to_tensor([[1], [2.]])
tf.constant([[1], [2.]])

a = tf.zeros([2, 2])
tf.zeros_like(a)
tf.zeros(a.shape)
tf.ones(1)
tf.ones([2, 3])
tf.fill([2, 3], 3)

tf.random.normal([2, 2], mean=0, stddev=1)
tf.random.truncated_normal([2, 2], mean=0, stddev=1)
tf.random.uniform([2, 2], minval=0, maxval=1)  # 均匀分布

index = tf.random.shuffle(tf.range(10))  # 按第一个维度打散
a = tf.random.normal([10, 784])
b = tf.random.uniform([10], maxval=10, dtype=tf.int32)
# 根据打散的index重新获取训练数据和标签数据
a = tf.gather(a, index)
b = tf.gather(b, index)

# LOSS计算
out = tf.random.uniform([4, 10])  # 4张照片 10类
y = tf.range(4)  # 假定第一张图片的lable是0 第二张是1 第三张是2 第四张是3
y = tf.one_hot(y, depth=10)
'''
 y --> 
    array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],...]
'''
# 计算 求和（y-out）^2 每个样本算一次
loss = tf.keras.losses.mse(y, out)
print(loss)
# 计算 样本平均
loss = tf.reduce_mean(loss)
print(loss)

# 网络拓扑
net = layers.Dense(10)
net.build((4, 8))
print(net.kernel)  # 参数w
print(net.bias)  # 参数b，默认初始化全为0的向量
# 例子 一个简单的网络拓扑
x = tf.random.normal([4, 784])
net = layers.Dense(10)
print(net(x).shape)  # 将x送到net最终得到的结果的形状
print(net.kernel.shape)  # 将x送到net 中间层的W矩阵
print(net.bias.shape)  # 将x送到net 中间层b的向量

# 将生活中的例子转换为tensor
# 三维的tensor -->影评文字
(X_train, y_train), (X_test, y_test) = \
    keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=80)
print(x_train.shape)
#emb = embedding(x_train)
#emb.shape #--> TensorShape([25000, 80, 100]) 25000个影评，每个影评80个单词，
          # 每个单词用100长度的向量编码

# 四维的tensor -->图片
# feature maps: [b,h,w,c] 第一个参数图片的索引，第二个高度 第三个宽度 第四个3->对应三元色
x = tf.random.normal((4, 32, 32, 3))

