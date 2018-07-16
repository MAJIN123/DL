# -*- coding: utf-8 -*-
# @Time    : 2018/7/16 15:53
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : country.py
# @Software: PyCharm

import json, time, codecs, re, sys

reload(sys)
sys.setdefaultencoding('utf-8')


def train_test_data(infile, outfile1, outfile2, str_):
    count = 0
    with codecs.open(infile, 'r', 'utf-8') as fr, \
            codecs.open(outfile1, 'rb', 'utf-8') as fr1, \
            codecs.open(outfile2, 'w', 'utf-8') as fw:

        countrySet = set([line.strip() for line in fr1])
        for line in fr:
            data = json.loads(line)
            for key, value in data.items():
                for i in range(6, len(value)):
                    flag = 0
                    for j in range(i + 1, len(value)):
                        if value[j][2] in str_:
                            flag = 1
                            break
                    if flag == 1:
                        break

                countryList = []
                for val in countrySet:
                    if val in value[2][2]:
                        countryList.append(val)
                        if len(set(countryList)) > 1:
                            break

                if len(countryList) == 1:
                    fw.write(value[2][2])
                    # print(countryList[0])
                    count += 1
                    if count % 3000 == 0:
                        print(value[2][2])
                        print(countryList[0])
    print(count)


if __name__ == '__main__':
    begin = time.time()

    json_file = '/Users/maloney/PycharmProjects/version_4/last_result_baike+finance_person_kg_intro+triple_update_filter_v4.txt'
    out_file1 = 'country_loc.txt'
    out_file2 = 'test.txt'
    out_file3 = 'lable_test.txt'
    str_list = [u'国家', u'国籍']

    # test_re()
    train_test_data(json_file, out_file1, out_file2, str_list)

    print ('The code took:', (time.time() - begin))
