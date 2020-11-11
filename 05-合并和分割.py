import tensorflow as tf

# 合并拼接
a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])
c = tf.concat([a, b], axis=0)
print(c.shape)

a = tf.ones([4, 32, 8])
b = tf.ones([4, 3, 8])
d = tf.concat([a, b], axis=1)
print(d.shape)

# 堆叠拼接 -->会创建一个新维度 形状必须相同
a = tf.ones([4, 32, 8])
b = tf.ones([4, 32, 8])
e = tf.stack([a, b], axis=1)
print(e.shape)
f, g = tf.unstack(e, axis=1)
print(f.shape, g.shape)


