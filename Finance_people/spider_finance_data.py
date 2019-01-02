# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 15:08
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


def get_links(soup):
    href = []
    fin_tab_1 = soup.find('div', class_=re.compile('title mt15'))
    if fin_tab_1:
        for link in fin_tab_1.find_all('a'):
            pure_link = link.get('href')
            # print(pure_link)
            if pure_link is not None:
                if 'http' in pure_link:
                    href.append(pure_link)
                else:
                    href.append('http://finance.people.com.cn' + pure_link)

    fin_tab_2 = soup.find('div', class_=re.compile('p1_content w1000'))
    if fin_tab_2:
        for link in fin_tab_2.find_all('a'):
            pure_link = link.get('href')
            # print(pure_link)
            if pure_link is not None:
                if 'http' in pure_link:
                    href.append(pure_link)
                else:
                    href.append('http://finance.people.com.cn' + pure_link)
    fin_tab_3 = soup.find('div', class_=re.compile('w1000 tbtj_box clearfix'))
    if fin_tab_3:
        for link in fin_tab_3.find_all('a'):
            pure_link = link.get('href')
            # print(pure_link)
            if pure_link is not None:
                if 'http' in pure_link:
                    pass
                else:
                    pure_link = 'http://finance.people.com.cn' + pure_link
            if pure_link:
                pure_link = pure_link.replace('\t', '').replace('\n', '')
            if pure_link not in href:
                href.append(pure_link)
    fin_tab_4 = soup.find('div', class_=re.compile('img_list1 clearfix'))
    if fin_tab_4:
        for link in fin_tab_4.find_all('a'):
            pure_link = link.get('href')
            # print(pure_link)
            if pure_link is not None:
                if 'http' in pure_link:
                    pass
                else:
                    pure_link = 'http://finance.people.com.cn' + pure_link
            if pure_link:
                pure_link = pure_link.replace('\t', '').replace('\n', '')
            if pure_link not in href:
                href.append(pure_link)
    fin_tab_5 = soup.find('div', class_=re.compile('w1000 mt20 column_2 p9_con'))
    if fin_tab_5:
        for link in fin_tab_5.find_all('a'):
            pure_link = link.get('href')
            # print(pure_link)
            if pure_link is not None:
                if 'http' in pure_link:
                    pass
                else:
                    pure_link = 'http://finance.people.com.cn' + pure_link
            if pure_link:
                pure_link = pure_link.replace('\t', '').replace('\n', '')
            if pure_link not in href:
                href.append(pure_link)

                # print(pure_link)

    return href


def get_text(soup):
    tag = soup.find('h1')
    tag_1 = soup.find('div', class_=re.compile('box_con'), id=re.compile('rwb_zw'))
    tag_2 = soup.find('div', class_=re.compile('content clear clearfix'))
    if tag and (tag_1 or tag_2):
        print('<title>' + tag.get_text().encode('utf8'))

    if tag_1:
        for t in tag_1.find_all('p'):
            # fout.write(t.get_text())
            text = t.get_text().encode('utf8')
            if text is not '\n':
                print(text.strip('\n'))

    if tag_2:
        for t in tag_2.find_all('p'):
            # fout.write(t.get_text())
            text = t.get_text().encode('utf8')
            if text is not '\n':
                print(text.strip('\n'))


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
url = 'http://finance.people.com.cn'
html_doc = get_html(url)
soup = BeautifulSoup(html_doc, 'html5lib')
link_list = get_links(soup)
#print(link_list, len(link_list))

#
# out_file = './text_data'
# fout = open(out_file, 'w+')

get_all_text(link_list)

end = time.time()
# print('the code took %s (s)' % int(end - start))
# print(len(link_list))
