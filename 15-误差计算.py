import tensorflow as tf
'''
mse --> 利用l2范数计算
cross entropy loss --> 交叉熵损失
    使用交叉熵的原因在于 
        其一：sigmoid+MSE相结合的时候容易出现梯度消失的情况，
              这样w和b会更新得非常慢
        其二：交叉熵收敛得更快
hinge loss --> 铰链损失函数
'''
tf.random.set_seed(42)
# mse
y = tf.constant([1, 2, 3, 0, 2])
y = tf.one_hot(y, depth=4)
y = tf.cast(y, dtype=tf.float32)

out = tf.random.normal([5, 4])

loss1 = tf.reduce_mean(tf.square(y-out))
loss2 = tf.square(tf.norm(y-out))/(5*4)
loss3 = tf.reduce_mean(tf.losses.MSE(y, out))

print(loss1)
print(loss2)
print(loss3)

# 熵 熵越大说明包含的信息量越大，10个苹果 5个坏的 5个好的 的信息量要大于
# 全部是坏的或者全部是好的 的情况，对应的熵也更大
a = tf.fill([4], 0.25)
# a = tf.math.log(a) / tf.math.log(2.) # -> log函数默认以e为底，通过换底公式变成以2为底
print(a)
print(tf.reduce_sum(a*(tf.math.log(a)/tf.math.log(2.))))

a = tf.constant([0.1, 0.1, 0.1, 0.7])
print(tf.reduce_sum(a*(tf.math.log(a)/tf.math.log(2.))))

a = tf.constant([0.01, 0.01, 0.01, 0.97])
print(tf.reduce_sum(a*(tf.math.log(a)/tf.math.log(2.))))

# 交叉熵 衡量数据集和数据集之间的相似程度
# https://www.bilibili.com/video/BV1Sp4y1a7cx?from=search&seid=15933807621956756738
# H(p,q) = 求和（p(x) log q(x)）
# 交叉熵及el散度其实就是一种最大似然，使用了对数将乘法变成了加法而已 (知乎上有解读文章)
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.25, 0.25, 0.25]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.5, 0., 0.25]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0.25, 0.75, 0., 0.]))
print(tf.losses.categorical_crossentropy([0, 1, 0, 0], [0, 1, 0., 0.]))
print(tf.losses.BinaryCrossentropy()([0, 1, 0, 0], [0, 1, 0., 0.])) # 另一种调用方式，基本相同
print(tf.losses.BinaryCrossentropy()([1], [0.7])) # 处理二分类
#使用from_logits等于true来增加稳定性
x = tf.random.normal([1, 784])
w = tf.random.normal([784, 2])
b = tf.zeros([2])
logits = x@w+b
print(logits.shape)
prob = tf.squeeze(tf.math.softmax(logits, axis=1))
print(tf.losses.categorical_crossentropy([0, 1], prob))
# softmax也一并的做了，当from_logits为true的时候，且避免了训练中数值不稳定的问题
print(tf.losses.categorical_crossentropy([0, 1], tf.squeeze(logits, axis=0), from_logits=True))


