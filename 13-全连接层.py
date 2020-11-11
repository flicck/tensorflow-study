from tensorflow import keras
import tensorflow as tf
'''
real deeplearning
matmul --> neural network --> deep learning --> multi-layer

recap h = relu(X@W+b)

深度学习为什么最近这些年发展得很快:
bigData,relu,dropOut,batchNorm,resNet,initialization,tensorflow
'''
# dense全连接 单层
# x = tf.random.normal([4, 784])
# net = tf.keras.layers.Dense(512)
# out = net(x) #其中自动调用了build
# print(net.kernel.shape) # [784, 512]
# print(net.bias.shape) # [512,]
# print(out.shape) # [4,512]
#
# net = tf.keras.layers.Dense(10)
# net.build(input_shape=(None, 4))
# print(net.kernel.shape)
# print(net.bias.shape)
# net.build(input_shape=(None, 20))
# print(net.kernel.shape)
# print(net.bias.shape)

# dense 多层
x = tf.random.normal([2, 3])
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3)
])
model.build(input_shape=[None, 3])
model.summary() # print的意思

for p in model.trainable_variables:
    print(p.name, p.shape)


