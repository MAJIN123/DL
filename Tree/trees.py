# -*- coding: utf-8 -*-
# @Time    : 2018/7/14 20:21
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : trees.py
# @Software: PyCharm

from math import log
import operator
import treePlotter


def calShannonEnt(dataSet):
    res = 0.0
    m = len(dataSet)
    labelCounts = {}
    for item in dataSet:
        curLabel = item[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 0
        labelCounts[curLabel] += 1
    for key in labelCounts:
        prob = float(labelCounts[key]) / m
        res -= prob * log(prob, 2)
    return res


def creatDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    returnDataSet = []
    for item in dataSet:
        if item[axis] == value:
            tp = item[:axis]
            tp.extend(item[axis + 1:])
            returnDataSet.append(tp)
    return returnDataSet


def chooseBestFeature2Split(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        uniValList = set([val[i] for val in dataSet])
        newEntropy = 0.0
        for val in uniValList:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for val in classList:
        if val not in classCount.keys():
            classCount[val] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]


def creatTree(dataSet, labels):
    classList = [val[-1] for val in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(dataSet)
    bestFeat = chooseBestFeature2Split(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    uniFeatVals = set([val[bestFeat] for val in dataSet])
    for val in uniFeatVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][val] = creatTree(splitDataSet(dataSet, bestFeat, val), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict:
        if key == testVec[featIndex]:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def loadTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


d, l = creatDataSet()
# treePlotter.createPlot(creatTree(d, l))
tree = (creatTree(d, l))
storeTree(tree, 'tree')
print(tree)
print(loadTree('tree'))
