import tensorflow as tf

'''
Sort/argsort
Topk
Top-5 Acc.
'''
# 排序
tf.random.set_seed(42)
a = tf.random.shuffle(tf.range(5))
print(a)
print(tf.sort(a, direction='DESCENDING'))  # 将元素排序
print(tf.argsort(a, direction='DESCENDING'))  # 获得元素排序后对应的原下标
idx = tf.argsort(a, direction='DESCENDING')
print(tf.gather(a, idx))  # 将元素按照所给的下标排序

# 返回top几
a = tf.convert_to_tensor([[4, 6, 8],
                          [9, 4, 7],
                          [4, 5, 1]])
res = tf.math.top_k(a, 2)
print(res.indices)  # 返回前2个在原矩阵的下标
print(res.values)  # 返回前2个在原矩阵的值
# 例: top-K accuracy的计算过程
def accuray(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)
    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)
    return res

output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('prob:', output.numpy())
pred = tf.argmax(output, axis=1)
print('pred:', pred.numpy())
print('label:', target.numpy())
acc = accuray(output, target, topk=(1, 2, 3, 4, 5, 6))
print('top-1-6 acc:', acc)
