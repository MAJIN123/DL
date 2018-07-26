# -*- coding: utf-8 -*-
# @Time    : 2018/7/25 16:13
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : crf.py
# @Software: PyCharm

import datetime
import random
import numpy as np
from collections import defaultdict
from scipy.misc import logsumexp
class dataSet(object):
    def __init__(self, filename):
        self.filename = filename
        self.sentnces = []
        self.tags = []
        sentence = []
        tag = []
        word_num = 0
        fr = open(filename)
        while True:
            line = fr.readline()
            if not line:
                break
            if line == '\n':
                self.sentnces.append(sentence)
                self.tags.append(tag)
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])
                tag.append(line.split()[3])
                word_num += 1
        self.NumSentnces = len(self.sentnces)
        self.NumTags = word_num

        print('%s有%d个句子以及%d个词。\n' % (filename, self.NumSentnces, self.NumTags))
        fr.close()

    def shuffle(self):
        tp = [(s, t) for s, t in zip(self.sentnces, self.tags)]
        random.shuffle(tp)
        self.sentnces = []
        self.tags = []
        for s, t in tp:
            self.sentnces.append(s)
            self.tags.append(t)


class CRF(object):
    def __init__(self, trainDataFile=None, devDataFile=None, testDataFile=None):
        self.trainData = dataSet(trainDataFile) if trainDataFile is not None else None
        self.devData = dataSet(devDataFile) if devDataFile is not None else None
        self.testData = dataSet(testDataFile) if testDataFile is not None else None
        self.features = {}
        self.weights = []
        self.tag2id = {}
        self.tags = []
        self.EOS = 'EOS'
        self.BOS = 'BOS'

    def creatBigramFeature(self, preTag, curTag):
        return ['01:' + curTag + '*' + preTag]

    def creatUnigramFeature(self, sentence, position, curTag):
        template = []
        curWord = sentence[position]
        curWordFirstChar = curWord[0]
        curWordLastChar = curWord[-1]

        if position == 0:
            lastWord = '##'
            lastWordLastChar = '#'
        else:
            lastWord = sentence[position - 1]
            lastWordLastChar = lastWord[-1]

        if position == len(sentence) - 1:
            nextWord = '$$'
            nextWordFirstChar = '$'
        else:
            nextWord = sentence[position + 1]
            nextWordFirstChar = nextWord[0]

        template.append('02:' + curTag + '*' + curWord)
        template.append('03:' + curTag + '*' + lastWord)
        template.append('04:' + curTag + '*' + nextWord)
        template.append('05:' + curTag + '*' + curWord + '*' + lastWordLastChar)
        template.append('06:' + curTag + '*' + curWord + '*' + nextWordFirstChar)
        template.append('07:' + curTag + '*' + curWordFirstChar)
        template.append('08:' + curTag + '*' + curWordLastChar)

        for i in range(len(sentence[position]) - 1):
            template.append('09:' + curTag + '*' + sentence[position][i])
            template.append('10:' + curTag + '*' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + curTag + '*' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + curTag + '*' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + curTag + '*' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append('12:' + curTag + '*' + curWord + '*' + lastWordLastChar + '*' + nextWordFirstChar)

        for i in range(4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + curTag + '*' + sentence[position][0:i + 1])
            template.append('15:' + curTag + '*' + sentence[position][-i - 1:])

        return template

    def creatFeatureTemplate(self, sentence, position, preTag, curTag):
        template = []
        template.extend(self.creatBigramFeature(preTag, curTag))
        template.extend(self.creatUnigramFeature(sentence, position, curTag))
        return template

    def creatFeatureSpace(self):
        for i in range(len(self.trainData.sentnces)):
            sentence = self.trainData.sentnces[i]
            tags = self.trainData.tags[i]
            for j in range(len(sentence)):
                if j == 0:
                    preTag = self.BOS
                else:
                    preTag = tags[j-1]
                template = self.creatFeatureTemplate(sentence,j,preTag,tags[j])
                for tp in template:
                    if tp not in self.features:
                        self.features = len(self.features)
                for tag in tags:
                    if tag not in self.tags:
                        self.tags.append(tag)
        self.tags = sorted(self.tags)
        self.tag2id = {t:i for i,t in enumerate(self.tags)}
        self.weights = np.zeros(len(self.features))
        self.g=defaultdict(float)
        self.bigramFeature = [
            [self.creatBigramFeature(preTag,tag) for preTag in self.tags]
            for tag in self.tags
        ]
        self.bigramScore = np.zeros(len(self.tags),len(self.tags))

        print("the total number of features is %d"% len(self.features))

    def score(self,feature):
        scores = [self.weights[self.features[f]] for f in feature if f in self.features]
        return sum(scores)

    def forward(self,sentence):
        pathScores = np.zeros(len(sentence),len(self.tags))

        pathScores[0] = [self.score(self.creatFeatureTemplate(sentence,0,self.BOS,tag)) for tag in self.tags]
        for i in range(1,len(sentence)):
            unigramScore = np.array([self.score(self.creatUnigramFeature(sentence,i,tag)) for tag in self.tag])
            scores = self.bigramScore + unigramScore[:,None]
            pathScores[i] = logsumexp(pathScores[i-1]+scores,axis = 1)

            return pathScores
    def backWard(self,sentence):
        pathScores = np.zeros(len(sentence),len(self.tags))

        for i in reversed(range(len(sentence)-1)):
            unigramScore = np.array([self.score(self.creatUnigramFeature(sentence,i+1,tag)) for tag in self.tags])
            scores = self.bigramScore.T+ unigramScore
            pathScores[i] = logsumexp(pathScores[i+1]+ scores,axis=1)

        return pathScores

    def updataGradient(self,sentence,tags):
        for i in range(len(sentence)):
            if i == 1:
                preTag = self.BOS
            else:
                preTag = tags[i-1]
            curTag = tags[i]
            feature = self.creatFeatureTemplate(sentence,i,preTag,curTag)
            for f in feature:
                if f in self.features:
                    self.g[self.features[f]] += 1

        alpha = self.forward(sentence)
        beta = self.backward(sentence)
        logZ = logsumexp(alpha[-1])


    def predict(self,sentence):
        NumOfWords = len(sentence)
        NumOfTags = len(self.tag2id)

        maxScore = np.zeros(NumOfWords,NumOfTags)
        paths = np.zeros((NumOfWords,NumOfTags),dtype='int')

        # for j in range(NumOfTags):


