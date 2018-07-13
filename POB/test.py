# -*- coding: utf-8 -*-
# @Time    : 2018/7/12 16:11
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : test.py
# @Software: PyCharm

import codecs
import json
import sys
import time
from itertools import izip

# sys.path.insert(0, '/Users/maloney/PycharmProjects/version_4/NER_SYS')
#
# from NER_SYS.NER_8E import *


sys.path.insert(0, '/data/jma/new_attr/NER_SYS')

from NER_8E import *


def run_limit(infile, infile_test, outfile):
    all_num = 0.0
    indenfiy_num = 0.0
    correct_num = 0.0
    with codecs.open(infile, 'r', 'utf-8') as fr, codecs.open(infile_test, 'r', 'utf-8') as f, codecs.open(outfile, 'w',
                                                                                                           'utf-8') as fw:
        for line1, line2 in izip(fr, f):
            temp = ner_er(line1.replace('[', '').replace(']', ''))['POB']
            if temp.find('[') >= 0:
                m = [x.replace('@POB', '') for x in re.findall(r'\[.*?@POB\]', temp)]
                n = re.findall(r'\[.*?\]', line2)[0]
                flag = 1
                for _ in m:
                    if n in _:
                        correct_num += 1

                        fw.write(temp)
                        flag = 0
                        break
                if flag == 1:
                    print('识别出错:')
                    print(temp.strip())
                    print(line2.strip())

                indenfiy_num += 1
            else:
                print('未识别:')
                print(temp.strip())
                print(line2.strip())
            all_num += 1
            # print(all_num)

        # fw3.write(json.dumps({key: value}, ensure_ascii=False) + '\n')
    print [correct_num, indenfiy_num, all_num]
    p = (correct_num * 1.0) / (indenfiy_num * 1.0)
    r = (correct_num * 1.0) / (all_num * 1.0)
    f = 2 * p * r / (p + r)
    print 'P:', "%.2f" % (p * 100)
    print 'R:', "%.2f" % (r * 100)
    print 'F:', "%.2f" % (f * 100)


if __name__ == '__main__':
    begin = time.time()

    input_file = 'data/test.txt'
    test_file = 'data/lable_test.txt'
    out_file = 'test-res_v2.txt'

    run_limit(input_file, test_file, out_file)

    print ('The code took:', (time.time() - begin))