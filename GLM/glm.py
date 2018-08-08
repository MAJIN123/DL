# -*- coding: utf-8 -*-
# @Time    : 2018/8/8 19:48
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : glm.py
# @Software: PyCharm

import numpy as np


class GloBalLinearModel(object):

    def __init__(self, nt):
        self.nt = nt

    def unigram(self, wordseq, index, ti):
        word = wordseq[index]
        prev_word = wordseq[index - 1] if index > 0 else '^^'
        next_word = wordseq[index + 1] if index < len(wordseq) - 1 else '$$'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('02', ti, word))
        fvector.append(('03', ti, prev_word))
        fvector.append(('04', ti, next_word))
        fvector.append(('05', ti, word, prev_char))
        fvector.append(('06', ti, word, next_char))
        fvector.append(('07', ti, first_char))
        fvector.append(('08', ti, last_char))

        for char in word[1: -1]:
            fvector.append(('09', ti, char))
            fvector.append(('10', ti, first_char, char))
            fvector.append(('11', ti, last_char, char))
        if len(word) == 1:
            fvector.append(('12', ti, word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                fvector.append(('13', ti, char, 'consecutive'))
            if i <= 4:
                fvector.append(('14', ti, word[: i]))
                fvector.append(('15', ti, word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', ti, word))
            fvector.append(('15', ti, word))
        return fvector

    def bigram(self, prev_ti, ti):
        return [('01', ti, prev_ti)]

    def create_feature_template(self, wordseq, index, prev_ti, ti):
        bigram = self.bigram(prev_ti, ti)
        unigram = self.unigram(wordseq, index, ti)
        return bigram + unigram

    def create_feature_space(self, data):
        self.feature_space = list({f for wordseq, tiseq in data for f in
                                   set(self.create_feature_template(wordseq, 0, -1, tiseq[0])).union(
                                       *[self.create_feature_template(wordseq, i, tiseq[i - 1], tiseq[i]) for i in
                                         range(1, len(tiseq))])})

        self.fdict = {f:i for i,f in enumerate(self.feature_space)}
        self.d = len(self.feature_space)
        self.W = np.zeros(self.d)
        self.V = np.zeros(self.d)
        self.BF = [[self.bigram(prev_ti,ti) for prev_ti in range(self.nt)] for ti in range(self.nt)]

    def score(self,fv,average=False):
        if average:
            scores = [self.V[self.fdict[f]] for f in fv if f in self.fdict]
        else:
            scores = [self.W[self.fdict[f]] for f in fv if f in self.fdict]
        return sum(scores)

    def predict(self,wordseq,average=False):




