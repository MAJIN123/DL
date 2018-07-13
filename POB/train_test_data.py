# -*- coding: utf-8 -*-
# @Time    : 2018/7/12 14:14
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : train_test_data.py
# @Software: PyCharm

import json, time, codecs, re, sys

reload(sys)
sys.setdefaultencoding('utf-8')


def train_test_data(infile, outfile1, outfile2, outfile3, str_,neg_list):
    count = 1
    with codecs.open(infile, 'r', 'utf-8') as fr, \
            codecs.open(outfile1, 'w', 'utf-8') as fw1, \
            codecs.open(outfile2, 'w', 'utf-8') as fw2, \
            codecs.open(outfile3, 'w', 'utf-8') as fw3:
        for line in fr:
            data = json.loads(line)
            for key, value in data.items():
                for i in range(6, len(value)):
                    if value[i][1] in str_ and value[i][2] is not None and value[i][2] not in [' ', '', u'中国']:
                        flag = 0
                        for j in range(i + 1, len(value)):
                            if value[j][2] in neg_list:
                                flag = 1
                                break
                        if flag == 1:
                            break
                        value[2][2] = value[2][2].replace('[', '').replace(']', '').replace('(', '').replace(')','').replace('（', '').replace('）', '')
                        if value[i][2].endswith(u'人'):
                            value[i][2] = value[i][2][0:-1]
                            print(value[i][2])

                        _r = u''.join(value[i][2].replace('(', '').replace(')', ''))
                        _p = '[' + value[i][2] + ']'

                        # _p = '[' + value[i][2] + ']'
                        # print(value[2][2])
                        # print(value[i][2])
                        try:
                            tem = re.sub(_r, _p, value[2][2])
                        except:
                            print('re error!')
                        else:
                            if len(tem) > len(value[2][2]):
                                # print(value[2][2])
                                # print(tem)
                                if count % 8 != 0:
                                    fw1.write(tem + '\n')
                                else:
                                    fw2.write(value[2][2] + '\n')
                                    fw3.write(tem + '\n')
                                count += 1
                        break

    print(count)


if __name__ == '__main__':
    begin = time.time()

    json_file = '/Users/maloney/PycharmProjects/version_4/last_result_baike+finance_person_kg_intro+triple_update_filter_v4.txt'
    out_file1 = 'data/modify/train_ori.txt'
    out_file2 = 'data/modify/test.txt'
    out_file3 = 'data/modify/lable_test.txt'
    str_list = [u'出生地', u'出生地点']
    negative_list = [u'出生', u'职业地点', u'居住地', u'现居地', u'工作地点', u'家乡', u'国籍', u'籍贯', u'原籍']

    # test_re()
    train_test_data(json_file, out_file1, out_file2, out_file3, str_list,negative_list)

    print ('The code took:', (time.time() - begin))
