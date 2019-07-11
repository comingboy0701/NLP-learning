# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:26:35 2019

@author: dell
"""
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
       reponse = requests.get(url, headers=headers,timeout=2 )
       print(reponse)
       if reponse.status_code == 200:
           return reponse.content
       return None
   except:
       return None

def get_sub_parse(html):
    html =  str(html.decode("utf-8"))
    pattern_url = re.compile(r'target=_blank href="(.*?)"')
    url_sub = re.findall(pattern_url, html)
    url_cn = [parse.unquote(url) for url in url_sub ]
    pattern_sub= re.compile(r'(北京地铁[\d+\u2E80-\u9FFF]{1,10}线)')
    subs= list(set(re.findall(pattern_sub,str(url_cn))))
    
    for sub in subs:
        if (sub not in produce_sub) and (sub not in consumer_sub):
            produce_sub.append(sub)

def write_sub(sub,stations=''):
    with open('station.csv','a+') as f:
        f.write(sub+',')
        for station in stations:
            f.write(station+',')
        f.write('\n')

def sucess_stations(html2):
    pattern_station = re.compile(r'>([\d+\u2E80-\u9FFF]{1,10}站)<')
    stations = re.findall(pattern_station, str(html2))
    if len(stations)>1:
        write_sub(sub,stations)
        print('获取地铁: *****%s站*****成功' %(str(sub)))
    else:
        pattern_station = re.compile(r'[\d+\u2E80-\u9FFF]{1,10}')
        stations = re.findall(pattern_station, str(html2))
        if len(stations)>1:
            write_sub(sub,stations)
            print('获取地铁: *****%s站*****成功' %(str(sub)))
        else:
            write_sub(sub,stations)
            print('只获取地铁线路: *****%s站*****需要明确' %(str(sub)))
  
def get_station_parse(sub,html):
    html =  str(html.decode("utf-8"))
    pattern_html2 = re.compile(r'<table.*?车站列表.*?车站名称(.*?)</table>',re.S) 
    html2 = re.findall(pattern_html2, html)
    if len(html2)>0 :
        sucess_stations(html2)
    else:
        pattern_html2 = re.compile(r'[(车站列表)|(车站信息)].*?<table.*?车站名称(.*?)</table>',re.S)
        html2 = re.findall(pattern_html2, html)
        if len(html2)>0:
            sucess_stations(html2)
        else:
            pattern_html2 = re.compile(r'站点.*?<table.*?中文站名(.*?)</table>',re.S)
            html2 = re.findall(pattern_html2, html)
            if len(html2)>0:
                sucess_stations(html2)
            else:
                pattern_html2 = re.compile(r'<div class="para" label-module="para">([\d+\u2E80-\u9FFF]{1,10})</div>',re.S)
                html2 = re.findall(pattern_html2, html)
                if len(html2)>0:
                    sucess_stations(html2)
                else:
                    write_sub(sub)
                    print('只获取地铁线路: *****%s站*****需要明确' %(str(sub)))

produce_sub = []
consumer_sub = []

url = 'https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
html = get_one_page(url)
get_sub_parse(html)

while produce_sub:
    print(len(produce_sub))
    sub = produce_sub.pop(0)
    url_main = 'https://baike.baidu.com/item/'
    url = url_main + quote(sub)
    html = get_one_page(url)
    if html:
        get_station_parse(sub,html)
        get_sub_parse(html)
        consumer_sub.append(sub)
    time.sleep(1)




    