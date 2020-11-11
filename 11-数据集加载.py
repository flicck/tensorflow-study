from tensorflow import keras
import tensorflow as tf
'''
>keras.datasets
>tf.data.Dataset.from_tensor_slices
>we will talk input pipeline later
'''
# keras.datasets
# boston housing,mnist/fashion mnist,cifar10/100,imdb
# mnist [28, 28, 1] 70k 60k 10k
# 60k       10k   ->目前还是numpy的ndarray
(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x.shape)
print(x.shape)
print(x.min(), x.max(), x.mean())
print(y[:4]) # y目前不是onehot结构
y_onehot = tf.one_hot(y, depth=10)
print(y_onehot[:2])

# cifar10/100 分类图片 对同样的图片 10指的是分成10类 100指的是分成100类
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x.shape, y.shape, x_test.shape, y_test.shape)
print(x.min(), x.max())
print(y[:4])
# 为什么要转成dataSet呢，因为dataSet提供了很多预处理操作，且支持多线程
db = tf.data.Dataset.from_tensor_slices(x_test).batch(10)
print(next(iter(db)).shape)
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
print(next(iter(db))[1].shape)
# 打散功能
db.shuffle(10000)
# 定义预处理函数
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.reshape(tf.one_hot(y, depth=10), [10, ])
    return x, y
# 将预处理函数设置进入db
db2 = db.map(preprocess)
print(next(iter(db2)))
# batch
db3 = db2.batch(32)
res = next(iter(db3))
print(res[0].shape, res[1].shape)
# 堆叠数据集 -->相当于epochs 多次迭代
db4 = db3.repeat(2)

# example:minist
def prepare_minist_features_and_lables(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y
def mnist_dataset():
    (x, y), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    y = tf.one_hot(y, depth=10)
    y_val = tf.one_hot(y_val, depth=10)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_minist_features_and_lables)
    ds = ds.shuffle(60000).batch(100)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(prepare_minist_features_and_lables)
    ds_val = ds_val.shuffle(10000).batch(100)
    return ds, ds_val




