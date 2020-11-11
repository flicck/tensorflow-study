import tensorflow as tf
'''
sigmoid  softmax tanh
sigmoid函数大量使用容易出现梯度消失问题：
    https://blog.csdn.net/feizxiang3/article/details/101294382
'''
# sigmoid 将数据变成0到1之间的数
a = tf.linspace(-6., 6., 10)
print(a)
print(tf.sigmoid(a)) #变成0到1之间的数

x = tf.random.normal([1, 28, 28]) * 5
print(x.shape)

x = tf.sigmoid(x)
print(tf.reduce_min(x), tf.reduce_max(x))

# softmax将数据变成一种加和为1的方式
a = tf.linspace(-2., 2., 5)
print(a)
x = tf.nn.softmax(a)
print(x)

logits = tf.random.uniform([1, 10], minval=-2, maxval=2)
print(logits)
prob = tf.nn.softmax(logits, axis=1)
print(prob)
print(tf.reduce_sum(prob, axis=1))

# tanh 将值映射到-1到1之间
a = tf.range(-2., 3., 1)
print(a)
print(tf.tanh(a))
