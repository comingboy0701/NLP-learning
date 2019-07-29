# encoding: utf-8 
""" 
@author: chen km 
@contact: 760855003@qq.com 
@file: test04.py 
@time: 2019/7/25 21:01 
"""
import random
import matplotlib.pylab as plt
from collections import defaultdict
import math

points = defaultdict(dict)
latitudes = [random.randint(-100, 100) for _ in range(10)]
longitude = [random.randint(-100, 100) for _ in range(10)]
for k ,la_lg in enumerate(zip(latitudes,longitude)):
    points[k] = la_lg



def points_distance(points):




x_1,y_1 = points[5]

plt.plot(latitudes, longitude,'bo')
plt.plot(x_1,y_1 ,'ro')
plt.show()

def distance(point_1,point_2):
    x_1,y_1= point_1
    x_2, y_2 = point_2
    return math.sqrt((x_1-x_2)^2+(y_1-y_2)^2)

def search(start, is_goal):
    distance(point_1, point_2)


dis = distance(points[0],points[1])


