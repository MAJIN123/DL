# -*- coding: utf-8 -*-
# @Time    : 2018/12/18 14:25
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : run.py
# @Software: PyCharm


#awk 'BEGIN{b=0} {if($0==""){b=1;next;} if(b){print "\n"$0;b=0;}else print;}' input >output

import os
import time

com = './run.sh'
while True:
    os.system(com)
    time.sleep(3600*24)