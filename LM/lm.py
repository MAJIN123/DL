# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 20:22
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : lm.py
# @Software: PyCharm

import datetime
import random
import numpy as np
from Config import config
import pickle
from collections import defaultdict
from scipy.misc import logsumexp


class dataSet(object):
    def __init__(self, filename):
        self.filename = filename
        self.sentences = []
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
                self.sentences.append(sentence)
                self.tags.append(tag)
                sentence = []
                tag = []
            else:
                sentence.append(line.split()[1])
                tag.append(line.split()[3])
                word_num += 1
        self.NumSentnces = len(self.sentences)
        self.NumTags = word_num

        print('%s有%d个句子以及%d个词。\n' % (filename, self.NumSentnces, self.NumTags))
        fr.close()

    def split(self):
        data = []
        for i in range(len(self.sentences)):
            for j in range(len(self.sentences[i])):
                data.append((self.sentences[i], j, self.tags[i][j]))
        return data


class liner_model(object):
    def __init__(self, trainDataFile=None, devDataFile=None, testDataFile=None):
        self.trainData = dataSet(trainDataFile) if trainDataFile is not None else None
        self.devData = dataSet(devDataFile) if devDataFile is not None else None
        self.testData = dataSet(testDataFile) if testDataFile is not None else None
        self.features = {}
        self.weights = []
        self.tags = []
        self.v = []
        self.tag2id = {}

    def create_feature_template(self, sentence, tag, position):
        template = []
        cur_word = sentence[position]
        cur_tag = tag
        cur_word_first_char = cur_word[0]
        cur_word_last_char = cur_word[-1]
        if position == 0:
            last_word = '##'
            last_word_last_char = '#'
        else:
            last_word = sentence[position - 1]
            last_word_last_char = sentence[position - 1][-1]

        if position == len(sentence) - 1:
            next_word = '$$'
            next_word_first_char = '$'
        else:
            next_word = sentence[position + 1]
            next_word_first_char = sentence[position + 1][0]

        template.append('02:' + cur_tag + '*' + cur_word)
        template.append('03:' + cur_tag + '*' + last_word)
        template.append('04:' + cur_tag + '*' + next_word)
        template.append('05:' + cur_tag + '*' + cur_word + '*' + last_word_last_char)
        template.append('06:' + cur_tag + '*' + cur_word + '*' + next_word_first_char)
        template.append('07:' + cur_tag + '*' + cur_word_first_char)
        template.append('08:' + cur_tag + '*' + cur_word_last_char)

        for i in range(1, len(sentence[position]) - 1):
            template.append('09:' + cur_tag + '*' + sentence[position][i])
            template.append('10:' + cur_tag + '*' + sentence[position][0] + '*' + sentence[position][i])
            template.append('11:' + cur_tag + '*' + sentence[position][-1] + '*' + sentence[position][i])
            if sentence[position][i] == sentence[position][i + 1]:
                template.append('13:' + cur_tag + '*' + sentence[position][i] + '*' + 'consecutive')

        if len(sentence[position]) > 1 and sentence[position][0] == sentence[position][1]:
            template.append('13:' + cur_tag + '*' + sentence[position][0] + '*' + 'consecutive')

        if len(sentence[position]) == 1:
            template.append(
                '12:' + cur_tag + '*' + cur_word + '*' + last_word_last_char + '*' + next_word_first_char)

        for i in range(0, 4):
            if i > len(sentence[position]) - 1:
                break
            template.append('14:' + cur_tag + '*' + sentence[position][0:i + 1])
            template.append('15:' + cur_tag + '*' + sentence[position][-(i + 1)::])

        return template

    def creat_feature_space(self):
        T = len(self.trainData.sentences)
        for i in range(T):
            sentence = self.trainData.sentences[i]
            tags = self.trainData.tags[i]
            for j in range(len(sentence)):
                templates = self.create_feature_template(sentence, tags[j], j)
                for f in templates:
                    if f not in self.features.keys():
                        self.features[f] = len(self.features)
                for tag in tags:
                    if tag not in self.tags:
                        self.tags.append(tag)

            self.weights = np.zeros(len(self.features), dtype='int32')
            self.v = np.zeros(len(self.features), dtype='int32')
            self.tags = sorted(self.tags)
            self.tag2id = {tag: id for id, tag in enumerate(self.tags)}
            print("the total number of features is %d" % (len(self.features)))

    def dot(self, feature, averaged=False):
        score = 0
        for f in feature:
            if f in self.features:
                if averaged == False:
                    score += self.weights[self.features[f]]
                else:
                    score += self.v[self.features[f]]
        return score

    def predict(self, sentence, position, averaged=False):
        tagid = np.argmax(
            [self.dot(self.create_feature_template(sentence, tag, position), averaged) for tag in self.tags])
        return self.tags[tagid]

    def evaluate(self, data, averaged=False):
        total_num, correct_num = 0, 0
        for i in range(len(data.sentences)):
            sentence = data.sentences[i]
            tags = data.tags[i]
            total_num += len(tags)
            for j in range(len(tags)):
                predict_tag = self.predict(sentence, j, averaged)
                if predict_tag == tags[j]:
                    correct_num += 1

        return (correct_num, total_num, correct_num / total_num)

    def online_train(self, iterator=20, averaged=False, shuffle=False, exitor=20):
        max_dev_p = 0
        max_iter = -1
        counter = 0
        data = self.trainData.split()

        if averaged == True:
            print('using v to predic dev data...')
        for iter in range(iterator):
            starttime = datetime.datetime.now()
            print('iterator:%d' % (iter))
            if shuffle == True:
                print('\tshuffle the train data...')
                random.shuffle(data)
            for i in range(len(data)):
                sentence = data[i][0]
                j = data[i][1]
                gold_tag = data[j][2]
                predict_tag = self.predict(sentence, j, averaged)
                if predict_tag != gold_tag:
                    feature_loss = self.create_feature_template(sentence, predict_tag, j)
                    feature_gold = self.create_feature_template(sentence, gold_tag, j)
                    for f in feature_loss:
                        if f in self.features.keys():
                            self.weights[self.features[f]] -= 1
                    for f in feature_gold:
                        if f in self.features.keys():
                            self.weights[self.features[f]] += 1
                    self.v += self.weights
            train_correct_num, total_num, train_precision = self.evaluate(self.trainData, averaged)
            print('\t' + 'train准确率：%d / %d = %f' % (train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.devData, averaged)
            print('\t' + 'dev准确率：%d / %d = %f' % (dev_correct_num, dev_num, dev_precision))

            if self.testData != None:
                test_correct_num, test_num, test_precision = self.evaluate(self.testData, averaged)
                print('\t' + 'test准确率：%d / %d = %f' % (test_correct_num, test_num, test_precision))

            if dev_precision > max_dev_p:
                max_dev_precision = dev_precision
                max_iterator = iter
                counter = 0
            else:
                counter += 1
                # self.save('./result.txt')

            endtime = datetime.datetime.now()
            print("\titeration executing time is " + str((endtime - starttime)) + " s")

            if train_correct_num == total_num:
                break
            if counter >= exitor:
                break
        print('iterator = %d , max_dev_precision = %f' % (max_iter, max_dev_p))

    def dump(self,file):
        with open(file,'w') as fw:
            pickle.dump(self,fw)

    @classmethod
    def load(cls,file):
        with open(file,'rb') as fr:
            lm = pickle.load(fr)
        return lm


if __name__ == '__main__':
    train_data_file = config['train_data_file']
    dev_data_file = config['dev_data_file']
    test_data_file = config['test_data_file']
    averaged = config['averaged']
    iterator = config['iterator']
    shuffle = config['shuffle']
    exitor = config['exitor']

    starttime = datetime.datetime.now()
    lm = liner_model(train_data_file, dev_data_file, test_data_file)
    lm.creat_feature_space()
    lm.online_train(iterator, averaged, shuffle, exitor)
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime)))
