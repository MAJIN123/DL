# -*- coding: utf-8 -*-
# @Time    : 2018/12/17 20:04
# @Author  : Maloney
# @Site    : jma@192.168.126.124
# @File    : spider_finance_data.py
# @Software: PyCharm

from bs4 import BeautifulSoup
import urllib
import requests, re, time


def get_html(url):
    html = requests.get(url).content
    return html


def get_fin_tabs(soup):
    href = []
    fin_tab_1 = soup.find('div', class_=re.compile('m-part m-part1 udv-clearfix'))
    if fin_tab_1:
        for link in fin_tab_1.find_all('a'):
            pure_link = link.get('href')
            # print(pure_link)
            if pure_link is not None:
                href.append(pure_link)

    fin_tab_2 = soup.find('div', class_=re.compile('m-part m-part2 udv-clearfix'))
    if fin_tab_2:
        for link in fin_tab_2.find_all('a'):
            pure_link = link.get('href')
            # print(pure_link)
            if pure_link is not None:
                href.append(pure_link)

                # print(pure_link)

    return href


def get_text(soup):
    tag = soup.find('h1')
    tag_1 = soup.find('div', class_=re.compile('article'), id=re.compile('artibody'))
    if tag and tag_1:
        print('<title>' + tag.get_text().encode('utf8'))

    if tag_1:
        for t in tag_1.find_all('p'):
            # fout.write(t.get_text())
            text = t.get_text().encode('utf8')
            if text is not '\n':
                print(text)


def get_all_text(link_list):
    for url in link_list:
        try:
            html_doc = get_html(url)
            soup = BeautifulSoup(html_doc, 'html5lib')
            get_text(soup)
            # print('\n')
        except:
            pass


start = time.time()
url = 'https://finance.sina.com.cn'
html_doc = get_html(url)
soup = BeautifulSoup(html_doc, 'html5lib')
link_list = get_fin_tabs(soup)
#
# out_file = './text_data'
# fout = open(out_file, 'w+')
get_all_text(link_list)
end = time.time()
# print('the code took %s (s)' % int(end - start))
# print(len(link_list))
