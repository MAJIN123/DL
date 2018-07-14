# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 14:23
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : kNN.py
# @Software: PyCharm

import numpy
import operator
import random
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
def creatDateSet():
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMate = diffMat ** 2
    sqDistances = sqDiffMate.sum(axis=1)
    distancess = sqDistances ** 0.5
    sortedDistIndicies = distancess.argsort()

    classCount = {}
    for i in range(k):
        label = labels[sortedDistIndicies[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename, 'r')
    lines = fr.readlines()
    lenOflines = len(lines)
    returnMat = numpy.zeros((lenOflines, 3))
    classLabelVector = []
    index = 0
    for line in lines:
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def writeData(filename):
    fw = open(filename, 'w')
    for i in range(1000):
        for x in range(3):
            tm = random.uniform(0, 10)
            print(tm)
            print(type(tm))
            fw.write(str(tm))
            fw.write('\t')
        t = random.randint(1, 3)
        fw.write(str(t))
        fw.write('\n')


def autoNorm(dataSet):
    minVals = numpy.min(dataSet, axis=0)
    # minVals = dataSet.max(axis=0)
    maxVals = numpy.max(dataSet, axis=0)
    ranges = maxVals - minVals

    normDataSet = numpy.zeros(numpy.shape(dataSet))
    m = numpy.shape(dataSet)[0]
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print errorCount


def img2vec(filename):
    returnMat = numpy.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnMat[0][32 * i + j] = line[j]
    return returnMat


def handwritingClassTest():
    hwLabels = []
    train_filelist = listdir('digits/trainingDigits')
    m = len(train_filelist)
    trainMat = numpy.zeros((m, 1024))
    for i in range(m):
        filename = train_filelist[i]
        filestr = filename.split('.')[0]
        hwLabels.append(int(filestr.split('_')[0]))
        trainMat[i, :] = img2vec('digits/trainingDigits/%s' % filename)

    test_filelist = listdir('digits/testDigits')
    errorCount = 0.0
    for i in range(len(test_filelist)):
        filename = test_filelist[i]
        filestr = filename.split('.')[0]
        label = int(filestr.split('_')[0])
        vecUnderTest = img2vec('digits/testDigits/%s' % filename)
        classifierRes = classify0(vecUnderTest, trainMat, hwLabels, 3)
        print "the classifier came back with : %d, the real answer is: %d" % (classifierRes, label)
        if classifierRes != label:
            errorCount += 1
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(len(test_filelist)))




# writeData('datingTestSet.txt')

# group, labels = creatDateSet()
#
# print(classify0([0, 0], group, labels, 3))

# datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
# print(type(datingDataMat))
# normMat, ranges, minVals = autoNorm(datingDataMat)
# # print(datingDataMat)
# # print(normMat)
# # print(ranges)
# # print(minVals)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.scatter(normMat[:, 1], normMat[:, 2], 15.0 * numpy.array(normMat), 15.0 * numpy.array(normMat))
# ax.scatter(normMat[:, 1], normMat[:, 2])
# plt.show()
# datingClassTest()
# print(datingLabels)
handwritingClassTest()