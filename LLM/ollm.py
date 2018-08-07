# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 14:27
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : ollm.py
# @Software: PyCharm

import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict
from scipy.misc import logsumexp
import pickle


class LoglinerModel(object):

    def __init__(self, nt):
        self.nt = nt

    def create_feature_space(self, data):
        self.epsilon = list({
            f for wordseq, tagseq in data
            for i, ti in enumerate(tagseq)
            for f in self.create_feature_template(wordseq, i)

        })
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        self.d = len(self.epsilon)
        self.W = np.zeros(self.d)

    def SGD(self, train, dev, file, epochs, batch_size, interval, eta, decay, lmbda, anneal, regularize, shuffle):
        update_n = 0
        total_time = timedelta()

        max_epoch, max_p = 0, 0.0
        iter_time = 0

        training_data = [(wordseq, i, ti) for wordseq, tagseq in train
                         for i, ti in enumerate(tagseq)]

        n = len(training_data)

        for epoch in range(epochs):
            iter_time += 1
            start = datetime.now()
            if shuffle:
                random.shuffle(training_data)
            if not regularize:
                lmbda = 0
            batches = [training_data[i:i + batch_size] for i in range(0, n, batch_size)]

            batch_n = len(batches)

            for batch in batches:
                if not anneal:
                    self.update(batch, lmbda, n, eta)
                else:
                    self.update(batch, lmbda, n, eta * decay ** (update_n / batch_n))
                update_n += 1

            print("Epoch %d / %d: " % (epoch, epochs))
            print("\ttrain: %d / %d = %4f" % self.evaluate(train))
            tp, total, p = self.evaluate(dev)
            print("\ttrain: %d / %d = %4f" % (tp, total, p))

            t = datetime.now() - start
            print("\t%ss elapsed" % t)
            total_time += t

            if p > max_p:
                max_epoch, max_p = epoch, p
                self.dump(file)
            elif epoch - max_epoch > interval:
                break

        print("max precision of dev is %4f at epoch %d" % (max_p, max_epoch))
        print("mean time of each is %s s" % (total_time / iter_time))

    def evaluate(self, data):
        tp, total = 0, 0

        for wordseq, tiseq in data:
            total += len(wordseq)
            piseq = np.array([self.predict(wordseq, i) for i in range(len(wordseq))])
            tp += np.sum(tiseq == piseq)

        return tp, total, tp / total

    def predict(self, wordseq, i):
        fv = self.create_feature_template(wordseq, i)
        scores = self.score(fv)
        return np.argmax(scores)

    def update(self, batch, lmbda, n, eta):
        gradients = defaultdict(float)

        for wordseq, i, ti in batch:
            fv = self.create_feature_template(wordseq, i)
            fiseq = [self.fdict[f] for f in fv if f in self.fdict]
            # for fi in fseq:
            #     gradients[fi] += 1

            # fvs = [self.create_feature_template(wordseq, i, ti) for ti in range(self.nt)]
            scores = self.score(fv)
            probs = np.exp(scores - logsumexp(scores))

            for fi in fiseq:
                gradients[fi, ti] += 1
                gradients[fi] -= probs

            if lmbda:
                self.W *= (1 - eta * lmbda / n)

            for k, v in gradients.items():
                self.W[k] += eta * v

    def dump(self, file):
        with open(file, 'wb') as fw:
            pickle.dump(self, fw)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as fr:
            ollm = pickle.load(fr)
        return ollm

    def score(self, fv):
        scores = np.array([self.W[self.fdict[f]] for f in fv if f in self.fdict])
        return np.sum(scores, axis=0)

    def create_feature_template(self, wordseq, i):
        word = wordseq[i]
        prev_word = wordseq[i - 1] if i > 0 else '^^'
        next_word = wordseq[i + 1] if i < len(wordseq) - 1 else '$$'
        prev_char = prev_word[-1]
        next_char = next_word[0]
        first_char = word[0]
        last_char = word[-1]

        fvector = []
        fvector.append(('02', word))
        fvector.append(('03', prev_word))
        fvector.append(('04', next_word))
        fvector.append(('05', word, prev_char))
        fvector.append(('06', word, next_char))
        fvector.append(('07', first_char))
        fvector.append(('08', last_char))

        for char in word[1:-1]:
            fvector.append(('09', char))
            fvector.append(('10', first_char, char))
            fvector.append(('11', last_char, char))
        if len(word) == 1:
            fvector.append(('12', word, prev_char, next_char))
        for i in range(1, len(word)):
            prev_char, char = word[i - 1], word[i]
            if prev_char == char:
                fvector.append(('13', char, 'consecutive'))
            if i <= 4:
                fvector.append(('14', word[:i]))
                fvector.append(('15', word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', word))
            fvector.append(('15', word))

        return fvector
