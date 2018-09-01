# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 20:39
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : dnn.py
# @Software: PyCharm

import numpy as np
import random
import mnist_loader
import time

import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforword(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def updata_mini_batch(self, mini_batch, eta):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            nable_b = [nb + dnb for nb, dnb in zip(nable_b, delta_b)]
            nable_w = [nw + dnw for nw, dnw in zip(nable_w, delta_w)]
        self.weights = [w - (eta / len(mini_batch) * nw) for w, nw in zip(self.weights, nable_w)]
        self.biases = [b - (eta / len(mini_batch) * nb) for b, nb in zip(self.biases, nable_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        x,X= [],[]
        plt.ion()
        plt.figure(1)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[i:i + mini_batch_size] for i in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updata_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1}/ {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

            x.append(j+1)
            X.append(self.loss(training_data))
            plt.plot(x,X)
            plt.draw()  # 注意此函数需要调用
            plt.pause(0.01)
            # time.sleep(0.01)
        # plt.show()

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforword(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def loss(self, data):
        # p_data = [[x,np.argmax(y)] for x,y in data]
        # for x, y in data:
        #     print((np.argmax(self.feedforword(x)),np.argmax(y)))
        results = [(np.argmax(self.feedforword(x)), np.argmax(y)) for x, y in data]
        return sum(int(x != y)**2 for (x, y) in results)/2

    def cost_derivative(self,output,y):
        return (output - y)

    def backprop(self, x, y):
        n_b = [np.zeros(b.shape) for b in self.biases]
        n_w = [np.zeros(w.shape) for w in self.weights]

        # feedforword
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backword pass
        delta = self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        n_b[-1]=delta
        n_w[-1]=np.dot(delta,activations[-2].T)

        for i in range(2,self.num_layers):
            z = zs[-i]
            delta = np.dot(self.weights[-i+1].T,delta)*sigmoid_prime(z)
            n_b[-i] = delta
            n_w[-i] = np.dot(delta,activations[-i-1].T)

        return (n_b,n_w)

train,val,test = mnist_loader.load_data_wrapper()
net = Network([784,100,10])
net.SGD(train,30,10,3.0,test)
# train =[(1,1)]
# net = Network([1,1,1])
# net.SGD(train,50,)


