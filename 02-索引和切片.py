import tensorflow as tf
import numpy as np
'''
切片
gather
gather_nd
'''
a = tf.random.normal([4, 28, 28, 3])
print(a[1, 2].shape)
print(a[1, 2, 3].shape)
# 第二张照片的第二行第三列的第2个通道
print(a[1, 2, 3, 2].shape)
print(a[-1:].shape)
print(a[-2:].shape)
print(a[:2].shape)
print(a[:-1].shape)
print(a[0, :, :, :].shape)
print(a[0, 1, :, :].shape)
# 双冒号是步长
print(a[:, ::2, ::2, :].shape)

# selective indexing
# tf.gather tf.gather_nd tf.boolean_mask
# axis 对哪一个维度进行抽样
a = tf.random.uniform([2, 4, 20, 3])  # 2个学校 4个班 20个同学 3个枯木
print(tf.gather(a, axis=0, indices=[2, 1, 3, 0]).shape)
print(tf.gather(a, axis=2, indices=[2, 1, 3, 0]).shape)
# gather_nd 和切片类似,但是更加灵活
print(tf.gather_nd(a, [0]).shape)
print(tf.gather_nd(a, [0, 1, 2]))
print(tf.gather_nd(a, [[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
