'''
keras:
    metric --->计量工具
        acc_meter = metrics.Accuracy()
        loss_meter = metrics.Mean()
        api:
            update_state
            result().numpy()
            reset_states
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


# 预处理
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)

batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(128)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(128)

db_iter = iter(db)
sample = next(db_iter)
print(sample[0].shape, sample[1].shape)

# 在容器中建立网络
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # 784 ->256
    layers.Dense(128, activation=tf.nn.relu),  # 256 ->128
    layers.Dense(64, activation=tf.nn.relu),  # 128 ->64
    layers.Dense(32, activation=tf.nn.relu),  # 64 ->32
    layers.Dense(10)  # 32 ->10
])
# 预创建
model.build(input_shape=[None, 28 * 28])
model.summary()
# w = w-lr*grad
optimizer = optimizers.Adam(lr=1e-3)

acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                # [b,784] [b,10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                prob = tf.nn.softmax(logits, axis=1)
                loss = tf.reduce_mean(tf.losses.MSE(y_onehot, prob))

                loss_meter.update_state(loss)

                loss2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            # 自动获得当前梯度
            grads = tape.gradient(loss, model.trainable_variables)
            # 原地更新梯度
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 9:
                print(step, float(loss2), float(loss))

                print('loss:',loss_meter.result().numpy())
                loss_meter.reset_states()

                # 拿测试集用于验证
                total_correct,total_num =0,0

                acc_meter.reset_states()
                
                for x,y in db_test:
                    x = tf.reshape(x, [-1, 28*28])
                    logits = model(x) # [b,10]
                    prob = tf.nn.softmax(logits, axis=1)
                    #b,10 ==> [b]
                    pred = tf.cast(tf.argmax(prob, axis=1),tf.int32)
                    # true,equa;
                    correct = tf.cast(tf.equal(pred,y),tf.int32)
                    corrct_num = tf.reduce_sum(correct)
                    total_correct += corrct_num
                    total_num += x.shape[0]
                    
                    acc_meter.update_state(y, pred)
                    
                acc =total_correct/total_num
                print(acc)
                print('成功率：',acc_meter.result().numpy())

if __name__ == '__main__':
    main()
