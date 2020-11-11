import tensorflow as tf
'''
梯度计算整体逻辑 一阶导 二阶导计算
常用激活函数求梯度 sigmoid tanh relu
常用损失函数求梯度 softmax+mse softmax+cross_entropy
'''
w = tf.constant(1.)
x = tf.constant(2.)
b = tf.constant(3.)
y = x*w

with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w
# y = x*w的计算并没有包裹到tape中，所以无法求得梯度
grad1 = tape.gradient(y, [w])
print(grad1)

with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x * w
# y2 = x*w的计算已包裹到tape中，所以可以求得梯度
grad2 = tape.gradient(y2, [w])
print(grad2)

# 进行可重复的计算梯度的操作
with tf.GradientTape(persistent=True) as tape:
    tape.watch([w])
    y3 = x * w

grad3 = tape.gradient(y3, [w])
print(grad3)
grad4 = tape.gradient(y3, [w])
print(grad4)


# 求二阶导数,使用嵌套
with tf.GradientTape() as t1:
    t1.watch([w])
    with tf.GradientTape() as t2:
        t2.watch([w, b])
        y4 = x * w * w * w + b
    dy_dw, dy_db = t2.gradient(y4, [w, b])
    print(dy_dw, dy_db)
d2y_dw2 = t1.gradient(dy_dw, w)
print(d2y_dw2)

# watch的方式也可以替换成Variable的方式放在外面
w = tf.Variable(1.0)
x = tf.Variable(2.0)
b = tf.Variable(3.0)
with tf.GradientTape() as t3:
    with tf.GradientTape() as t4:
        y5 = x * w * w * w + b
    dy_dw, dy_db = t4.gradient(y5, [w, b])
d2y_dw2 = t3.gradient(dy_dw, w)
print(dy_dw, dy_db)
print(d2y_dw2)

# sigmoid函数求梯度 0到1之间
a = tf.Variable(tf.linspace(-10., 10., 10))
print(a)
with tf.GradientTape() as tape:
    y = tf.sigmoid(a)
grads = tape.gradient(y, [a])
print(grads)

# tanh  -1到1之间 等于 2sigmoid(2x)-1
a = tf.Variable(tf.linspace(-10., 10., 10))
print(a)
with tf.GradientTape() as tape:
    y = tf.tanh(a)
grads = tape.gradient(y, [a])
print(grads)

# relu函数 小于0是梯度为0,leaky_relu小于0时仍有0.2的梯度
a = tf.Variable(tf.linspace(-10., 10., 10))
print(a)
with tf.GradientTape() as tape:
    y = tf.nn.relu(a)
grads = tape.gradient(y, [a])
print(grads)

a = tf.Variable(tf.linspace(-10., 10., 10))
print(a)
with tf.GradientTape() as tape:
    y = tf.nn.leaky_relu(a)
grads = tape.gradient(y, [a])
print(grads)

# mse求梯度softmax+mse
x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.zeros([3])
y = tf.constant([2, 0])
with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x@w+b, axis=1)
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))
grads = tape.gradient(loss, [w, b])
print(grads[0], grads[1])

# 交叉熵求梯度cross_entropy
with tf.GradientTape() as tape:
    tape.watch([w, b])
    logits = x@w+b
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y, depth=3), logits,
                                                            from_logits=True))
grads = tape.gradient(loss, [w,b])
print(grads[0], grads[1])




