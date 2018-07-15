# -*- coding: utf-8 -*-
# @Time    : 2018/7/15 19:37
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : test.py
# @Software: PyCharm

import trees
import treePlotter

fr = open('lenses.txt')
lenses = [line.strip().split('\t') for line in fr]
lenseLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lenseTree = trees.creatTree(lenses, lenseLabels)
treePlotter.createPlot(lenseTree)
