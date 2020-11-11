import tensorflow as tf
'''
pad 填充
tile 复制
'''
# 填充
a = tf.reshape(tf.range(9), [3, 3])
print(a)
print(tf.pad(a, [[1, 0], [1, 1]])) # 分别对应第一和第二个维度前后填充
#例子 填充图像
a = tf.random.normal([4, 28, 28, 3])
b = tf.pad(a, [[0, 0], [2, 2], [2, 2], [0, 0]]) # 相当于在每张图片的上下左右pad2个

# 维度复制
print(a)
print(tf.tile(a, [1, 2])) # 1表示当前这个维度不复制，2表示当前这个维度复制1次



