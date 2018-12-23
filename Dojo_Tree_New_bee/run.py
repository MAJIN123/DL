# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 15:31
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : run.py
# @Software: PyCharm
from itertools import combinations
import sys
import time
import math


def Dojo_extra_mark(in_file):
    fr = open(in_file, 'r')

    m_n_w_list = (fr.readline()).strip().split(',')
    m = int(m_n_w_list[0])
    n = int(m_n_w_list[1])
    w = int(m_n_w_list[2])

    rect_list = []
    for i in range(m):
        for j in range(n):
            rect_list.append([i, j])

    # 当资源数 小于 sqrt(m*n) 删除四个顶点
    flag = 0
    if w < math.sqrt(m * n):
        rect_list.remove([0, 0])
        rect_list.remove([m - 1, 0])
        rect_list.remove([0, n - 1])
        rect_list.remove([m - 1, n - 1])
        flag = 1

    t_list = list(combinations(rect_list, w))

    min_ = sys.maxsize
    res = []

    for choice in t_list:
        # 当资源数 小于 sqrt(m*n)，排除相互距离==1的候选答案
        if flag == 1:
            f = 0
            for c1 in choice:
                for c2 in choice:

                    if lenth_(c1, c2) == 1:
                        f = 1
                        continue
                if f == 1:
                    continue
            if f == 1:
                continue

        all_len = 0
        for point in rect_list:
            if point in choice:
                continue
            all_len += min_lenth(choice, point, m, n)
            if all_len > m * n - w:
                continue
        ## 当资源数 大于 sqrt(m*n) 结果必定< m *n - w
        if all_len < min_:
            min_ = all_len
            res = choice

        if min_ == m * n - w:
            break

    print('one solution: ', res)
    fr.close()


def min_lenth(choice, point, m, n):
    min = m + n
    for i in choice:
        len_ = lenth_(i, point)
        if len_ < min:
            min = len_
    return min


def lenth_(point_1, point_2):
    return abs(point_1[0] - point_2[0]) + abs(point_1[1] - point_2[1])


begin = time.time()
Dojo_extra_mark('in')
print('\nthe code took %s s' % (time.time() - begin))
