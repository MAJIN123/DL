#! /bin/bash

ls_date=`date +%Y-%m-%d`
file='./data/'${ls_date}
touch ${file}

nohup python spider_finance_data.py > ${file} 2>&1 &
