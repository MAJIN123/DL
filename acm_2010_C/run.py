# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 23:08
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : run.py
# @Software: PyCharm
import time


def C_Tracking_Bio_bots(in_file, out_file):
    fr = open(in_file, 'r')
    fw = open(out_file, 'w')

    nCase = 1

    while (True):
        m_n_w_list = (fr.readline()).strip().split()
        m = int(m_n_w_list[0])
        n = int(m_n_w_list[1])
        w = int(m_n_w_list[2])
        if m == 0 and n == 0 and w == 0:
            break

        wallArea = 0

        walls_list = []  # 放墙的list
        for i in range(w):
            _ = (fr.readline()).strip().split()
            x1 = int(_[0])
            y1 = int(_[1])
            x2 = int(_[2]) + 1
            y2 = int(_[3]) + 1
            new_wall = Rect(x1, x2, y1, y2)
            walls_list.append(new_wall)
            wallArea += new_wall.area()

        right_wall = Rect(n, n + 1, 0, m)
        top_wall = Rect(0, n, m, m + 1)
        walls_list.append(right_wall)
        walls_list.append(top_wall)
        wallArea += m + n

        idx = 0
        while idx < len(walls_list):
            cur_wall = walls_list[idx]

            for i in range(idx):
                old_wall = walls_list[i]
                if (cur_wall == right_wall and old_wall == top_wall) or (
                        cur_wall == top_wall and old_wall == right_wall):
                    continue
                if cur_wall.xmin < old_wall.xmin and old_wall.xmin <= cur_wall.xmax:
                    if old_wall.ymin < cur_wall.ymin and cur_wall.ymin <= old_wall.ymax:
                        walls_list.append(Rect(cur_wall.xmin, old_wall.xmin, cur_wall.ymin, old_wall.ymin))
                if old_wall.xmin < cur_wall.xmin and cur_wall.xmin <= old_wall.xmax:
                    if cur_wall.ymin < old_wall.ymin and old_wall.ymin <= cur_wall.ymax:
                        walls_list.append(Rect(old_wall.xmin, cur_wall.xmin, old_wall.ymin, cur_wall.ymin))
            idx += 1
        new_wall_area = union_(walls_list)
        print(wallArea)
        print(new_wall_area)
        fw.write("Case %d：%d\n" % (nCase, new_wall_area - wallArea))
        nCase += 1
    fw.close()
    fr.close()


def union_(walls):
    n_walls = walls

    area = 0
    if len(n_walls) == 0:
        area = 0
    elif len(n_walls) == 1:
        area = n_walls[0].area()
    else:
        taken = n_walls[0]
        del n_walls[0]
        area = taken.area()

        if area != 0:
            new_walls = []
            for i in range(len(n_walls)):
                new_wall = n_walls[i].intersect(taken)
                if new_wall != None:
                    new_walls.append(new_wall)
            area += union_(n_walls)
            area -= union_(new_walls)

    return area


class Rect(object):
    def __init__(self, x_1, x_2, y_1, y_2):
        self.xmin = min(x_1, x_2)
        self.xmax = max(x_1, x_2)
        self.ymin = min(y_1, y_2)
        self.ymax = max(y_1, y_2)

    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def intersect(self, r):
        if r.xmin >= self.xmax or self.xmin >= r.xmax or r.ymin >= self.ymax or self.ymin >= r.ymax:
            return None
        return Rect(max(self.xmin, r.xmin), min(self.xmax, r.xmax), max(self.ymin, r.ymin), min(self.ymax, r.ymax))


if __name__ == '__main__':
    begin = time.time()

    C_Tracking_Bio_bots('sample', 'C_out')

    print ('The code took:', (time.time() - begin)*10000)
