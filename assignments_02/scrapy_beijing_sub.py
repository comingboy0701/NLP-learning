# -*- coding: utf-8 -*-

import threading
import requests
from lxml import etree
import time
from fake_useragent import UserAgent
import random
import re

url ='https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'

def get_one_url(url):
   try:
       ua = UserAgent()
       headers = {'User-Agent': ua.random}
       reponse = requests.get(url, headers=headers, timeout=2)
       if reponse.status_code == 200:
           return reponse.content
       return None
   except:
       return None



def get_one_parse(url):
    html = str(get_one_url(url).decode("utf-8"))
    print(html)
    pattern_url = re.compile(r'target="_blank" href="(.*?)">北京地铁', re.S)

    url_sub = re.findall(pattern_url, html)

    sub =
    stations =

    print(html)



def write(sub):



def main()

    get_url()



if __name__ == '__main__':
    main()
