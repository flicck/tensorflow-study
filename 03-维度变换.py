import tensorflow as tf

# 更改形状 reshape
a = tf.random.normal([4, 28, 28, 3])
print(a.shape)
print(a.ndim)
print(tf.reshape(a, [4, 784, 3]).shape)
print(tf.reshape(a, [4, -1, 3]).shape) # 4 784 3
print(tf.reshape(a, [4, 784*3]).shape)
print(tf.reshape(a, [4, -1]).shape)
print(tf.reshape(tf.reshape(a, [4, -1]), [4, 14, 56, 3]))

# 调换维度 transpose
a = tf.random.normal((4, 3, 2, 1))
print(tf.transpose(a).shape) # [1, 2, 3, 4]
print(tf.transpose(a, perm=[0, 1, 3, 2]).shape) # [4, 3, 1, 2]

# 增加维度 expand_dims
a = tf.random.normal([4, 35, 8])
print(tf.expand_dims(a, axis=0).shape)
print(tf.expand_dims(a, axis=1).shape)

# 自动扩展维度以完成运算 Broadcasting
# broadcasting通常在计算过程中就自动完成，故不给出实例
