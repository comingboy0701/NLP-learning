# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:31:25 2019

@author: dell
"""
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


with open('station_2.csv','r') as f:
    location_cn = f.readlines()
    sub_stations = {}
    for sub in location_cn:
        sub_station = sub.strip().split(',')
        sub_stations[sub_station[0]] = sub_station[1:-1]

station_connection = defaultdict(list)    

for key,value in sub_stations.items():
    station_connection[value[0]].append(value[1].strip())
    station_connection[value[-1]].append(value[-2].strip())
    for index,station in enumerate(value[1:-1]):
        last_station = value[index].strip()
        station_connection[station].append(last_station)     
        next_station = value[index+2].strip()
        station_connection[station].append(next_station)
     
for key, value in station_connection.items():
    station_connection[key] = list(set(value))

        
with open('station_lati_1.csv','r') as f:
    stations_read = f.readlines()
    stations_location = {}
    for stations in stations_read:
        station = stations.strip().split(':')
        station_cn = station[0].split()[1]
        lati = tuple(map(float, station[1].split(',')))
        stations_location[station_cn] = lati
        print(station_cn,lati)


plt.figure(1)

stations_new = {}
for key, value in station_connection.items():
    stations_new[key] = stations_location[key]
cities = list(stations_new.keys())
city_graph = nx.Graph()
city_graph.add_nodes_from(cities)   
nx.draw(city_graph, stations_new, with_labels=False, node_size=10)


station_connection_graph = nx.Graph(station_connection)

plt.figure(2)
nx.draw(station_connection_graph, with_labels=False, node_size=10)


def geo_distance(origin, destination):
   
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def get_station_distance(station1, station2):
    return geo_distance(stations_location[station1], stations_location[station2])




def search (station1,station2,connnection_grpah,sort_candidate):
    pathes = [[station1]]
    vistied = set()
    while pathes:
        path = pathes.pop(0)
        froninter = path[-1]
        if froninter in vistied: continue
        successors = connnection_grpah[froninter]
        for station in successors:
            if station in path:continue
            new_path = path + [station]
            pathes.append(new_path)
            if station == station2: return new_path
        pathes = sort_candidate(pathes)
        vistied.add(froninter)       
        

def pretty_print(stations):
    print('>>'.join(stations))
    
def transfer_station_first(pathes):
    return sorted(pathes,key= len)


def get_path_distance(path):
    distance = 0
    for index,station in enumerate(path[:-1]):
        distance +=get_station_distance(station,path[index+1])
    return distance

def shortest_path_first(pathes):
    return sorted(pathes,key = get_path_distance)        

station1 = '天安门西站'
station2 = '奥体中心站'
# 最少换乘
stations = search(station1,station2,station_connection,sort_candidate=transfer_station_first)
print(pretty_print(stations))
# 最短距离
stations = search(station1,station2,station_connection,sort_candidate=shortest_path_first)
print(pretty_print(stations))



def is_goal(desitination):
    def _wrap(current_path):
        return current_path[-1] == desitination
    return _wrap


def search(graph, start, is_goal, search_strategy):
    pathes = [[start] ]
    seen = set()
    
    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]
        
        if froniter in seen: continue
            
        successors = graph[froniter]
        
        for city in successors: 
            if city in path: continue
            
            new_path = path+[city]
            
            pathes.append(new_path)
        
            if is_goal(new_path): return new_path
#        print('len(pathes)={}'.format(pathes))
        seen.add(froniter)
        pathes = search_strategy(pathes)
    
search(station_connection, start='奥体中心站', is_goal=is_goal('天安门西站'), search_strategy=lambda n: n)
