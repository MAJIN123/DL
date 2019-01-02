# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 22:37
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : run.py
# @Software: PyCharm


import os
import time

com = './run.sh'
while True:
    os.system(com)
    time.sleep(3600*24)