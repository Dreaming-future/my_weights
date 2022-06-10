#!/usr/bin/env python

import pickle
import gzip
import numpy as np

def load_data():
    """
    返回训练数据、验证数据和测试数据tuple
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    ``training_data``: 50,000 2-tuples ``(x, y)``
    ``x``: 输入图像 - 784-dimensional numpy.ndarray
    ``y``: 数字标签 - 10-dimensional numpy.ndarray
    ``validation_data``/``test_data``: 10,000 2-tuples ``(x, y)``
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    one-hot转换函数: 
    将数字(0...9)转为one-hot向量
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
