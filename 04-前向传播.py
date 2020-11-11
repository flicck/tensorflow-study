# out = relu(relu(relu(X@W1+B1)@W2+B2)@W3+B3)
# pred = argmax(out)
# loss = MSE(out, label)
# minimize loss -->] W1 W2 W3 B1 B2 B3

import tensorflow as tf
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), _ = datasets.mnist.load_data()

x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

#创建数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print(sample[0].shape, sample[1].shape)

# [b, 784] => [b, 512] => [b, 128] => [b, 10]
#         [784, 512]   [512, 128]   [128, 10]
# 也可以将b1视为w0，这样的话只要将w1增加一列，对train_db的x增加为1的一列就好
w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))
w2 = tf.Variable(tf.random.truncated_normal([512, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1

for epoch in range(10):
    for step, (x, y) in enumerate(train_db):
        # x:[128, 28, 28] ->[128,28*28]
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape: # 自动计算梯度-->跟踪tf.Variable类型的变量
            # y:[128]
            # h1 = x@w1 + b1 加号会自动进行broadcast
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # [b, 256] => [b. 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # [b, 128] => [b, 10]
            out = h2 @ w3 + b3

            # compute loss
            # out:[b, 10]
            # y:[b] => [b ,10]
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y - out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        print("==before==")
        for g in grads:
            print(tf.norm(g))

        grads, _ = tf.clip_by_global_norm(grads, 15) #限制梯度向量的范数不能超过15，超过的话会将其等比例缩小

        print("==after==")
        for g in grads:
            print(tf.norm(g))


        # w1 = w1 - learning_rate * w1_grad 必须使用assign进行原地更新
        # 否则会从variable包装变成原tensor
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        # 错误示范
        # w1 = w1 - lr * grads[0]
        # b1 = b1 - lr * grads[1]
        # w2 = w2 - lr * grads[2]
        # b2 = b2 - lr * grads[3]
        # w3 = w3 - lr * grads[4]
        # b3 = b3 - lr * grads[5]

        # 如果loss为nan了 为梯度消失或爆炸，可以降低标准差
        if step % 100 == 0:
            print(step, 'loss:', float(loss))
