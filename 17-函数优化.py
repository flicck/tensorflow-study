import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
# 定义函数
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

#获得散点
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range：', x.shape, y.shape)
x, y = np.meshgrid(x, y)
print('x, y maps:',x.shape, y.shape)
z = himmelblau([x, y])

#进行绘图
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z)
ax.view_init(30, -20)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#使用梯度下降进行优化
x = tf.constant([-4., 0.])
for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
        print(y)
    grads = tape.gradient(y, [x])
    print(grads)
    x -= 0.01 * tf.squeeze(tf.convert_to_tensor(grads), axis=0)
