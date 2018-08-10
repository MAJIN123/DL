# -*- coding: utf-8 -*-
# @Time    : 2018/8/8 20:05
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : run.py
# @Software: PyCharm
import numpy as np
from collections import Counter
a = [[1,23,4,6],[1]]
b = [[5,23,7,6],[1]]

# print(set(a).union(set(b)))
# aa = np.zeros((2,1)) # 列
# bb = np.zeros((2,2)) # 2*2
# cc = [1,1]
# aa[1,0] = 1
# print(aa)
# print(aa.reshape(1,-1))
# print(bb)
# print(bb.reshape(-1,1))
# print(cc)
# print(bb+aa+cc)
#
# test = [1,1,2,3,4,5,5,6,6,6,6]
# print(Counter(test))
# x,y=zip(*[(1,3),(2,5),(3,6)])
# print(x)
# print(y)
# ss,gg=map(list,zip(*[(1,3),(2,5),(3,6)]))
# print(ss)

npd = np.zeros(5,dtype='int')
w = np.zeros(5,dtype='int') + 1
print(w)
x,y = zip(npd,w)
print(x)
print(y)
