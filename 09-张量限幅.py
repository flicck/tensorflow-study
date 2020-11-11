import tensorflow as tf
'''
 clip_by_value
 relu
 clip_by_norm
 gradient clipping 常用
'''
# 两端限幅
a = tf.range(0, 10)
print(a)
print(tf.maximum(a, 2)) # 有比2大的就取大的，比2小就取2
print(tf.minimum(a, 8))
print(tf.clip_by_value(a, 2, 8)) # 限制矩阵在2到8的范围内

# relu限幅
a = a - 5
print(tf.nn.relu(a))
print(tf.maximum(a, 0))

# l2范数限幅 -->通过改变向量的模来实现
a = tf.random.normal([2, 2], mean=10)
print(tf.norm(a))
aa = tf.clip_by_norm(a, 15)
print(tf.norm(aa))

# 梯度限幅
# 为了解决梯度爆炸和梯度消失问题
# 见 04-前向传播中将学习率设置为1，使用clip_by_global_norm方法仍不会出现梯度爆炸情况
