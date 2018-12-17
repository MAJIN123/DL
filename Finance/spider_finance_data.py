# -*- coding: utf-8 -*-
# @Time    : 2018/12/17 20:04
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : spider_finance_data.py
# @Software: PyCharm

from bs4 import BeautifulSoup
import urllib
import requests,re

def get_html(url):
    html = requests.get(url).content
    return html

def get_fin_tabs(soup):
    href = []
    fin_tab_1 = soup.find('div',class_=re.compile('m-part m-part1 udv-clearfix'))
    if fin_tab_1:
        for link in fin_tab_1.find_all('a'):
            pure_link = link.get('href')
            #print(pure_link)
            if pure_link is not None:
                href.append(pure_link)

    fin_tab_2 = soup.find('div',class_=re.compile('m-part m-part2 udv-clearfix'))
    if fin_tab_2:
        for link in fin_tab_2.find_all('a'):
            pure_link = link.get('href')
            #print(pure_link)
            if pure_link is not None:
                href.append(pure_link)
                #print(pure_link)

    return href


url = 'https://finance.sina.com.cn'
html_doc = get_html(url)
#print(html_doc)
soup = BeautifulSoup(html_doc,'html5lib')
link_list = get_fin_tabs(soup)
print(len(link_list))

