#!/usr/bin/env python

"""
MLP网络+交叉熵损失函数+SGD优化器
"""

#### 导入库
import json
import random
import sys
import numpy as np
import seaborn

#### Miscellaneous functions
def vectorized_result(j):
    """ one-hot函数 """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """sigmoid导数"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """sigmoid导数"""
    return sigmoid(z)*(1-sigmoid(z))

#### MSE损失函数
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """计算网络前馈输出``a``和期望输出``y``间的损失 """
        return 0.5*np.linalg.norm(a-y)**2 # linear algebra: 0.5*

    @staticmethod
    def delta(z, a, y):
        """返回输出层的误差delta"""
        return (a-y) * sigmoid_prime(z)


#### 交叉熵损失函数
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        计算网络前馈输出``a``和期望输出``y``间的损失.
        np.nan_to_num保证数值计算稳定性.
        若``a``和``y``均为1, 则(1-y)*np.log(1-a) = nan
        np.nan_to_num保证此时输出为正确值0.
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """
        返回输出层的误差delta.
        参数``z``未使用, 只是为保证函数的接口一致性.
        """
        return (a-y)


#### Network类
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        列表``sizes``指定了网络每层神经元数量
        如列表[2, 3, 1]表示一个3层网络, 第一层2个神经元,
        第二层3个神经元, 第三层1个神经元.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initializer()
        self.cost=cost

    def weight_initializer(self):
        """
        权值初始化为N(0,1)标准正态分布
        偏置初始化为N(0,1)标准正态分布

        通常第一层为输入层, 神经元无偏置
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """获取神经网络关于输入``a``的输出."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):
        """
        使用mini-batch SGD优化
        ``training_data``: tuple list ``(x, y)``
        ``evaluation_data``: validation/test数据
        返回4 tuple lists:
        the (per-epoch) costs on the evaluation data
        the accuracies on the evaluation data
        the costs on the training data
        the accuracies on the training data.
        """

        # 早停功能
        best_accuracy=1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # 早停功能
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            #if monitor_training_cost:
            cost = self.total_cost(training_data, lmbda)
            training_cost.append(cost)
            print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # 早停
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        使用BP计算每个mini-batch的梯度, 并通过GD更新权值和偏置
        ``mini_batch``: tuple list ``(x, y)``
        ``eta``: 学习率
        ``lmbda``: 正则化参数
        ``n``: 训练集大小
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        返回tuple ``(nabla_b, nabla_w)``表示代价函数C_x的梯度
        ``nabla_b``和``nabla_w``是逐层list
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前馈
        activation = x
        activations = [x] # 逐层保存全部激活值
        zs = [] # 逐层保存全部z矢量值
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 后向传播 
        # equation 1
        # delta^(L) = dC/da * sigma'(z^(L))
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1表示最后一层神经元
        # l = 2表示导数第二层神经元

        # equation 2
        # delta^(l) = (w^(l+1)delta^(l+1)) * sigma'(z^(l))
        # equation 3
        # dC/db = delta^(l)
        # equation 4
        # dC/dw = delta^(l) * a^(l-1)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        返回输入``data``中神经网络输出正确结果的数量
        神经网络的输出是最后一层神经元中最大激活值的索引值
        ``convert``:
                False - validation/test data
                True - training data
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False): 
        """
        数据集``data``的全部损失
        ``convert``:
                False - training data
                True - validation/test data
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - power函数
        return cost

    def save(self, filename):
        """保存神经网络结构, 权值, 偏置和代价函数``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### 加载网络
def load(filename):
    """从文件filename加载网络，返回网络实例."""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net