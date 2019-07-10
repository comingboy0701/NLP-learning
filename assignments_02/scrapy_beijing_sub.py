# -*- coding: utf-8 -*-

import threading
import requests
from lxml import etree
import time
from fake_useragent import UserAgent
import random
import re
from urllib import parse
from urllib.request import quote


def get_one_page(url):
   try:
       ua = UserAgent()
       headers = {'User-Agent': ua.random}
       IP = {'http':random.choice(ips)} #指定对应的 IP 进行访问网址
       reponse = requests.get(url, headers=headers, proxies=IP,timeout=2 )
       print(reponse)
       if reponse.status_code == 200:
           return reponse.content
       return None
   except:
       return None


def get_sub_parse(url):
   
    html = str(get_one_page(url).decode("utf-8"))
    pattern_url = re.compile(r'target=_blank href="(.*?)"')
    url_sub = re.findall(pattern_url, html)
    url_cn = [parse.unquote(url) for url in url_sub ]
    url_cn_rm = list(set([url for url in url_cn if '线' in url]))
    sub_cn = list(set([re.findall(r'item/(.*?线)',i)[0] for i in url_cn_rm]))
    return url_cn_rm


def get_station_parse(url):
    html = 
    
    
def read_ip():
    with open(r'.\ip.txt','r') as f:
       lines = f.readlines()
       # 我们去掉lines每一项后面的\n\r之类的空格
       # 生成一个新的列表！
       ips = list(map(lambda x:x.strip(),[line for line in lines]))
       

def main():
    
    read_ip()
    
    url ='https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
    
    get_url = get_one_parse(url)

    get_url()



if __name__ == '__main__':
    main()
