#!/usr/bin/env python

# ----------------------
import mnist_loader
import network
import matplotlib.pyplot as plt

# 读取输入MNIST数据集
# 划分训练集、验证集、测试集
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# 构建3层MLP网络
# 代价函数为交叉熵
net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)

# SGD梯度下降
epoches = 300
[evaluation_cost, evaluation_accuracy, training_cost, training_accuracy] = \
net.SGD(training_data, epoches, 10, 0.1, lmbda = 5.0,
    evaluation_data=validation_data,
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)

# evaluation_cost = range(0,epoches,1)
# evaluation_accuracy = range(0,epoches,1)
# training_cost = range(0,epoches,1)
# training_accuracy = range(0,epoches,1)

t = range(0, epoches, 1)

plt.rcParams['figure.constrained_layout.use'] = True

ax1 = plt.subplot(221)
ax1.set_ylabel('Evaluation Cost')
plt.plot(t, evaluation_cost)

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
ax2.set_ylabel('Evaluation Accuracy')
plt.plot(t, evaluation_accuracy)

ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
ax3.set_ylabel('Training Cost')
plt.plot(t, training_cost)

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)
ax4.set_ylabel('Training Accuracy')
plt.plot(t, training_accuracy)

plt.show()

net.save("mnist_mlp.json")