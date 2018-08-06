# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 10:38
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : corpus.py
# @Software: PyCharm

import numpy as np


class Corpus(object):
    UNK = '<UNK>'

    def __init__(self, fdata):
        self.sentences = self.preprocess(fdata)
        self.wordseqs, self.tagseqs = zip(*self.sentences)
        self.words = sorted(set(np.hstack(self.wordseqs)))
        self.tags = sorted(set(np.hstack(self.tagseqs)))
        self.chars = sorted({c for w in self.words for c in w})
        self.chars.append(self.UNK)

        self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        self.ui = self.cdict[self.UNK]
        self.ns = len(self.sentences)
        self.nw = len(self.words)
        self.nt = len(self.tags)

    def load(self, fdata):
        data = []
        sentences = self.preprocess(fdata)

        for wordseq, tagseq in sentences:
            wordidseq = [
                tuple(self.cdict[c] if c in self.cdict else self.ui for c in w)
                for w in wordseq]
            tagidseq = [
                self.tdict[tag]
                for tag in tagseq]
            data.append((wordidseq, tagidseq))

        return data

    def size(self):
        return self.nw - 1, self.nt

    @staticmethod
    def preprocess(fdata):
        start = 0
        sentences = []

        with open(fdata, 'r') as fr:
            lines = [line for line in fr]
        c = len(lines)
        for i, line in enumerate(lines):
            if len(line) <= 1:
                splits = [l.split()[1:4:2] for l in lines[start:i]]
                wordseq, tagseq = zip(*splits)
                start = i + 1
                while start < c and len(lines[start]) <= 1:
                    start += 1
                sentences.append((wordseq, tagseq))

        return sentences
