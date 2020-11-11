import tensorflow as tf
'''
tf.norm 张量范数 tf.reduce_min/max 最小值最大值
tf.argmax/argmin 最大值最小值的位置
tf.equal
tf.unique
注意axis的含义，
    使用0值表示沿着每一列或行标签\索引值向下执行方法
    使用1值表示沿着每一行或者列标签模向执行对应的方法
    也就是说axis是哪个维度，返回的结果对应的维度就会消失，因为已经被聚合运算了
'''
# 张量范数
a = tf.ones([2, 2])
print(tf.norm(a)) # 默认l2范数 (2^2+2^2)^(1/2)
print(tf.sqrt(tf.reduce_sum(tf.square(a)))) # 默认l2范数
print(tf.norm(a, ord=1)) # l1范数->直接相加
print(tf.norm(a, ord=1, axis=0)) # 按行聚合求范数

#最大值、最小值、均值
a = tf.random.normal([4, 10])
print(tf.reduce_min(a))
print(tf.reduce_max(a))
print(tf.reduce_mean(a, axis=1)) # 按列聚合求均值

# 位置
print(a)
print(tf.argmax(a).shape) # 按行聚合，只剩列
print(tf.argmin(a, axis=1).shape) # 按列聚合，只剩行
tf.range(5)

# 比较
a = tf.constant([1, 2, 3, 2, 5])
b = tf.range(5) #[0, 1, 2, 3, 4]
res = tf.equal(a, b)
print(res) # 返回[false,false,false,false,false]
print(tf.cast(res, dtype=tf.int32)) #返回[0,0,0,0,0]

# 将值按大小重新按0开始的顺序给值
a = tf.constant([4, 2, 2, 4, 3])
print(tf.unique(a)) # [0, 1, 1, 0, 2]





