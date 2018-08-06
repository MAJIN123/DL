# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 11:14
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : run.py
# @Software: PyCharm

import corpus
import numpy as np

# c = corpus.Corpus('data/dev.conll')
# print(type(c.wordseqs))

a = ((1, 2, 3), (4, 5, 6), (1, 8, 9))
print(np.hstack(a))
print(set(np.hstack(a)))
print(sorted(set(a)))

l = list(i + j for i in range(10)
         for j in range(10))
print(l)
l = list({i + j for i in range(10)
          for j in range(10)})
print(l)
