# -*- coding: utf-8 -*-
# @Time    : 2018/8/25 15:16
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : logRegres.py
# @Software: PyCharm
import numpy as np


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    f = open(filename)
    for line in f.readlines():
        lineList = line.strip().split()
        dataMat.append([1.0, float(lineList[0]), float(lineList[1])])
        labelMat.append(int(lineList[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataIn, labelIn):
    dataMatrix = np.mat(dataIn)
    labelMat = np.mat(labelIn).T
    m, n = np.shape(dataMatrix)
    # print(dataMatrix)
    # print(labelMat)
    alpha = 0.0001
    maxIter = 10000
    weights = np.ones((n, 1))
    # print(weights)
    for i in range(maxIter):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.T * error
    return weights


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    d, l = loadDataSet('testSet.txt')
    dataArry = np.array(d)
    n = np.shape(dataArry)[0]


print(gradAscent(d, l))
