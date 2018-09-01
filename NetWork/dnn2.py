# -*- coding: utf-8 -*-
# @Time    : 2018/8/30 21:16
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : dnn2.py
# @Software: PyCharm

import numpy as np
import random
import mnist_loader
import matplotlib.pyplot as plt
import json
import sys
import time


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y)
        n_b[-1] = delta
        n_w[-1] = np.dot(delta, activations[-2].T)

        for j in xrange(2, self.num_layers):
            z = zs[-j]
            delta = np.dot(self.weights[-j + 1].T, delta) * sigmoid_prime(z)
            n_b[-j] = delta
            n_w[-j] = np.dot(delta, activations[-j - 1].T)

        return (n_b, n_w)

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            d_b, d_w = self.backprop(x, y)
            n_b = [nb + dnb for nb, dnb in zip(n_b, d_b)]
            n_w = [nw + dnw for nw, dnw in zip(n_w, d_w)]

        # L2
        self.weights = [(1 - lmbda * eta / n) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, n_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, n_b)]

    def SGD(self, train, epochs, mini_batch_size, eta, lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        train_cost, train_accuracy = [], []
        evaluation_cost, evaluation_accuracy = [], []

        if evaluation_data: n_data = len(evaluation_data)
        n = len(train)

        x, X = [], []
        # plt.ion()
        # plt.title(str(eta) + 'eta')
        # plt.figure(1)

        for j in xrange(epochs):
            random.shuffle(train)
            mini_batches = [train[i:i + mini_batch_size] for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(train))
            print("Epoch {0} training complete".format(j))
            if monitor_training_cost:
                cost = self.total_cost(train, lmbda)
                train_cost.append(cost)

                x.append(j + 1)
                X.append(cost)
                # plt.plot(x, X, '-r')
                # plt.draw()  # 注意此函数需要调用
                # # plt.pause(0.01)
                # time.sleep(0.01)

                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(train, convert=True)
                train_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data)

            print
        return x, X

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        n = len(data)
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / n
        cost += 0.5 * lmbda * sum(np.linalg.norm(w) ** 2
                                  for w in self.weights) / n
        # cost /= len(data)
        return cost

    def accuracy(self, data, convert=False):
        if convert:
            result = [(np.argmax(self.feedforward(x)), np.argmax(y))
                      for x, y in data]
        else:
            result = [(np.argmax(self.feedforward(x)), y)
                      for x, y in data]
        return sum(int(x == y) for (x, y) in result)

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1
    return e


train, val, test = mnist_loader.load_data_wrapper()

net1 = Network([784, 30, 10])
x, X = net1.SGD(train[:1000], 30, 10, 0.25, lmbda=2.5,
                evaluation_data=val,
                monitor_evaluation_cost=False,
                monitor_evaluation_accuracy=False,
                monitor_training_cost=True,
                monitor_training_accuracy=False)
# plt.ion()
plt.title('eta')
plt.plot(x, X, 'c-o', label='eta:0.25', linewidth=1, color='red')
# plt.show()

net2 = Network([784, 30, 10])
b, B = net2.SGD(train[:1000], 30, 10, 0.5, lmbda=2.5,
                evaluation_data=val,
                monitor_evaluation_cost=False,
                monitor_evaluation_accuracy=False,
                monitor_training_cost=True,
                monitor_training_accuracy=False)

plt.plot(b, B, 'r-o', label='eta:0.5', linewidth=1, color='blue')
# plt.show()

net3 = Network([784, 30, 10])
c, C = net3.SGD(train[:1000], 30, 10, 2.5, lmbda=2.5,
                evaluation_data=val,
                monitor_evaluation_cost=False,
                monitor_evaluation_accuracy=False,
                monitor_training_cost=True,
                monitor_training_accuracy=False)

plt.plot(c, C, 'b-o', label='eta:2.5', linewidth=1, color='black')
plt.show()
