# -*- coding: utf-8 -*-


import requests
import time
from fake_useragent import UserAgent
import re
from urllib import parse
from urllib.request import quote


def get_one_page(url):
   try:
       ua = UserAgent()
       headers = {'User-Agent': ua.random}
#       IP = {'http':random.choice(ips)} #指定对应的 IP 进行访问网址
       reponse = requests.get(url, headers=headers,timeout=2 )
       print(reponse)
       if reponse.status_code == 200:
           return reponse.content
       return None
   except:
       return None


def get_station_parse(url):
    url_main = 'https://baike.baidu.com/item/'
    url = url_main + quote(url)
    html =  str(get_one_page(url).decode("utf-8"))
    pattern_html2 = re.compile(r'车站列表.*?车站名称(.*?)table',re.S)
    
    html2 = re.findall(pattern_html2, html)
    
    pattern_station = re.compile(r'>([\u2E80-\u9FFF]{1,10}站)<')
    stations = re.findall(pattern_station, str(html2))
    
    write_sub(url,stations)
    
    return stations

def write_sub(sub,stations):
    with open('station.csv','a+') as f:
        f.write(sub+',')
        for station in stations:
            f.write(station+',')
        f.write('\n')
    
    
url_fisrt ='https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
html = str(get_one_page(url_fisrt).decode("utf-8"))
pattern_url = re.compile(r'target=_blank href="(.*?)"')
url_sub = re.findall(pattern_url, html)
url_cn = [parse.unquote(url) for url in url_sub ]
url_cn_rm = list(set([url for url in url_cn if '线' in url]))
sub_cn = list(set([re.findall(r'item/(.*?线)',i)[0] for i in url_cn_rm]))
for cn in sub_cn:
    get_station_parse(cn)
    time.sleep(1)

