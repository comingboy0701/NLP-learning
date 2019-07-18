# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:30:27 2019

@author: dell
"""

import requests
from lxml import etree
import time 

def get_loction(address):
    address = '北京地铁亦庄线万源街站'
    key = '76ca4ffd4960f784b41d5580e1680f52'
    url = 'https://restapi.amap.com/v3/geocode/geo?address={0}&output=XML&key={1}'.format(address,key)
    try:
        reponse = requests.get(url,timeout=2 )
        txt = etree.HTML(reponse.content)
        location  =txt.xpath("//location/text()")[0]
        return location
    except:
        return None
    
with open('station_1.csv','r') as f:
    location_cn = f.readlines()
    sub_stations = []
    for sub in location_cn:
        sub_station = sub.strip().split(',')
        sub,stations = sub_station[0],sub_station[1:-1]
        for station in stations:
            address = ' '.join([sub,station])
            sub_stations.append(address)
 
    
station_lati = {}     
while sub_stations:
    station = sub_stations.pop(0)
    address = ''.join(station.split())
    location = get_loction(address)
    if location:
        station_lati[station] = str(location)
        print(station,location)
    else:
        sub_stations.append(station)
    time.sleep(1)


with open('station_lati.csv','a') as f:
    for key,value in station_lati.items():
        f.write(key +':'+value)
        f.write('\n')
