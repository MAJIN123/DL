# -*- coding: utf-8 -*-
# @Time    : 2018/8/8 19:48
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : glm.py
# @Software: PyCharm

import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import random
import pickle


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

        self.fdict = {f: i for i, f in enumerate(self.feature_space)}
        self.d = len(self.feature_space)
        self.W = np.zeros(self.d)
        self.V = np.zeros(self.d)
        self.BF = [[self.bigram(prev_ti, ti) for prev_ti in range(self.nt)] for ti in range(self.nt)]

    def score(self, fv, average=False):
        if average:
            scores = [self.V[self.fdict[f]] for f in fv if f in self.fdict]
        else:
            scores = [self.W[self.fdict[f]] for f in fv if f in self.fdict]
        return sum(scores)

    def predict(self, wordseq, average=False):
        T = len(wordseq)
        delta = np.zeros(T, self.nt)
        paths = np.zeros(T, self.nt)
        bscores = np.array([self.score(bfv, average) for bfv in bfvs] for bfvs in self.BF)
        fvs = [self.create_feature_template(wordseq, 0, -1, ti) for ti in range(self.nt)]
        delta[0] = [self.score(fv, average) for fv in fvs]

        for i in range(1, T):
            uscore = np.array([self.score(self.unigram(wordseq, i, ti), average)] for ti in range(self.nt)).reshape(-1,
                                                                                                                    1)
            score = bscores + uscore + delta[i - 1]
            paths[i] = np.argmax(score, axis=1)
            delta[i] = score[np.arange(self.nt), paths[i]]

        prev = np.argmax(delta[-1])
        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        predict.reverse()

        return predict

    def update(self, batch):
        wordseq, tiseq = batch
        piseq = self.predict(wordseq)
        if not np.array_equal(tiseq, piseq):
            prev_ti, prev_pi = -1, -1
            for i, (ti, pi) in enumerate(zip(tiseq, piseq)):
                c = Counter(self.create_feature_template(wordseq, i, prev_ti, ti))
                e = Counter(self.create_feature_template(wordseq, i, prev_pi, pi))
                fiseq, fcounts = map(list, zip(*[(self.fdict[f], c[f] - e[f]) for f in c | e if f in self.fdict]))
                self.V[fiseq] += (self.K - self.R[fiseq]) * self.W[fiseq]
                self.W[fiseq] += fcounts
                self.R[fiseq] = self.K
                prev_ti, prev_pi = ti, pi
            self.K += 1

    def evaluate(self, data, average=False):
        tp, total, precision = 0, 0, 0.0

        for wordseq, tiseq in data:
            total += len(wordseq)
            piseq = np.array(self.predict(wordseq, average))
            tp += np.sum(tiseq == piseq)

        precision = float(tp) / total
        return tp, total, precision

    def online(self, train, dev, file, epochs, interval, average, shuffle):
        total_time = timedelta
        max_e, max_p = 0, 0.0

        for epoch in range(epochs):
            start = datetime.now()
            if shuffle:
                random.shuffle(train)
            self.K, self.R = 0, np.zeros(self.d, dtype='int')
            for batch in train:
                self.update(batch)
            self.V += [(self.K - r) * w for r, w in zip(self.R, self.W)]

            print("Epoch %d / %d: " % (epoch, epochs))
            result = self.evaluate(train, average=average)
            print("\ttrain: %d / %d = %4f" % result)
            tp, total, p = self.evaluate(dev, average=average)
            print("\tdev: %d / %d = %4f" % (tp, total, p))
            t = datetime.now() - start
            print("\t%ss elapsed" % t)
            total_time += t

            # 保存效果最好的模型
            if p > max_p:
                self.dump(file)
                max_e, max_p = epoch, p
            elif epoch - max_e > interval:
                break
        print("max precision of dev is %4f at epoch %d" % (max_p, max_e))
        print("mean time of each epoch is %ss" % (total_time / (epoch + 1)))

    def dump(self, file):
        with open(file, 'wb') as fw:
            pickle.dump(self, fw)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as fr:
            glm = pickle.load(fr)
        return glm
