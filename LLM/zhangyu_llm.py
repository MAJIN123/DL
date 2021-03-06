# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 21:25
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : zhangyu_llm.py
# @Software: PyCharm

import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.misc import logsumexp


class LogLinearModel(object):

    def __init__(self, nt):
        # 词性数量
        self.nt = nt

    def create_feature_space(self, data):
        # 特征空间
        self.epsilon = list({
            f for wordseq, tiseq in data
            for i, ti in enumerate(tiseq)
            for f in self.instantiate(wordseq, i, ti)
        })
        # 特征对应索引的字典
        self.fdict = {f: i for i, f in enumerate(self.epsilon)}
        # 特征空间维度
        self.d = len(self.epsilon)

        # 特征权重
        self.W = np.zeros(self.d)

    def SGD(self, train, dev, file,
            epochs, batch_size, interval, eta, decay, lmbda,
            anneal, regularize, shuffle):
        # 记录更新次数
        count = 0
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_precision = 0, 0.0

        training_data = [(wordseq, i, ti)
                         for wordseq, tiseq in train
                         for i, ti in enumerate(tiseq)]
        # 训练集大小
        n = len(training_data)
        # 迭代指定次数训练模型
        for epoch in range(epochs):
            start = datetime.now()
            # 随机打乱数据
            if shuffle:
                random.shuffle(training_data)
            # 设置L2正则化系数
            if not regularize:
                lmbda = 0
            # 按照指定大小对数据分割批次
            batches = [training_data[i:i + batch_size]
                       for i in range(0, n, batch_size)]
            nb = len(batches)
            # 根据批次数据更新权重
            for batch in batches:
                if not anneal:
                    self.update(batch, lmbda, n, eta)
                # 设置学习速率的指数衰减
                else:
                    self.update(batch, lmbda, n, eta * decay ** (count / nb))
                count += 1

            print("Epoch %d / %d: " % (epoch, epochs))
            print("\ttrain: %d / %d = %4f" % self.evaluate(train))
            tp, total, precision = self.evaluate(dev)
            print("\tdev: %d / %d = %4f" % (tp, total, precision))
            t = datetime.now() - start
            print("\t%ss elapsed" % t)
            total_time += t

            # 保存效果最好的模型
            if precision > max_precision:
                self.dump(file)
                max_e, max_precision = epoch, precision
            elif epoch - max_e > interval:
                break
        print("max precision of dev is %4f at epoch %d" %
              (max_precision, max_e))
        print("mean time of each epoch is %ss" % (total_time / (epoch + 1)))

    def update(self, batch, lmbda, n, eta):
        gradients = defaultdict(float)

        for wordseq, i, ti in batch:
            fv = self.instantiate(wordseq, i, ti)
            fiseq = (self.fdict[f] for f in fv if f in self.fdict)
            for fi in fiseq:
                gradients[fi] += 1

            # 获取每个词性对应的所有特征
            fvs = [self.instantiate(wordseq, i, ti) for ti in range(self.nt)]
            scores = np.array([self.score(fv) for fv in fvs])
            probs = np.exp(scores - logsumexp(scores))

            for fv, p in zip(fvs, probs):
                fiseq = (self.fdict[f] for f in fv if f in self.fdict)
                for fi in fiseq:
                    gradients[fi] -= p

        if lmbda != 0:
            self.W *= (1 - eta * lmbda / n)
        for k, v in gradients.items():
            self.W[k] += eta * v

    def predict(self, wordseq, index):
        fvs = [self.instantiate(wordseq, index, ti)
               for ti in range(self.nt)]
        scores = np.array([self.score(fv) for fv in fvs])
        return np.argmax(scores)

    def score(self, fvector):
        scores = [self.W[self.fdict[f]]
                  for f in fvector if f in self.fdict]
        return sum(scores)

    def instantiate(self, wordseq, index, ti):
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

        for char in word[1:-1]:
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
                fvector.append(('14', ti, word[:i]))
                fvector.append(('15', ti, word[-i:]))
        if len(word) <= 4:
            fvector.append(('14', ti, word))
            fvector.append(('15', ti, word))
        return fvector

    def evaluate(self, data):
        tp, total = 0, 0

        for wordseq, tiseq in data:
            total += len(wordseq)
            piseq = np.array([self.predict(wordseq, i)
                              for i in range(len(wordseq))])
            tp += np.sum(tiseq == piseq)
        precision = float(tp) / total
        return tp, total, precision

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            llm = pickle.load(f)
        return llm