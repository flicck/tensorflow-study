import tensorflow as tf
from matplotlib import pyplot as plt
'''
where
scatter_nd
meshgrid
'''
# where基础用法
# 返回True元素所在的坐标
# where配合gather可以拿到具体的值
tf.random.set_seed(42)
a = tf.random.normal([3, 3])
print(a)
mask = a > 0
print(mask)
print(tf.boolean_mask(a, mask)) # 取到位true的对应值
indices = tf.where(mask) # 获得true的坐标
print(tf.gather_nd(a, indices)) #通过gather获得对应值

# where高级用法-筛选
print(mask)
a = tf.ones([3, 3])
b = tf.zeros([3, 3])
print(tf.where(mask, a, b)) # 如果mask为true的话，从a中取，如果为false的话从b中取

# scatter_nd
# 根据indice对全为0的底板进行更新
shape = tf.constant([8])
indices = tf.constant([[4], [3], [1], [7]]) # 更新 4位置 3位置 1位置 7位置
updates = tf.constant([9, 10, 11, 12])
print(tf.scatter_nd(indices, updates, shape))

# meshgrid生成散点的工具
def func(x):
    # ...指的是略过所有维度
    z = tf.math.sin(x[..., 0]) + tf.math.cos(x[..., 1])
    return z
x = tf.linspace(0., 2 * 3.14, 500)
y = tf.linspace(0., 2 * 3.14, 500)
point_x, point_y = tf.meshgrid(x, y)
points = tf.stack([point_x, point_y], axis=2)
print(points.shape)
z = func(points)
print(z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(point_x, point_y, z)
plt.colorbar()
plt.show()