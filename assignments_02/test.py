# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:24:26 2019

@author: dell
"""
from collections import defaultdict

def bfs(graph, start):
    visited = [start]
    seen = set()
    while visited:
        froninter = visited.pop(0)
        if froninter in seen:continue
        for successor in graph[froninter]:
            if successor in seen:continue
            print(successor)
            visited.append(successor)
        seen.add(froninter)
    return seen



def dfs(graph, start):
    visited = [start]
    seen = set()
    while visited:
        froninter = visited.pop(-1)
        if froninter in seen:continue
        for successor in graph[froninter]:
            if successor in seen:continue
            print(successor)
            visited.append(successor)
        seen.add(froninter)
    return seen

graph = {0:[1,2],
         1:[3,4],
         2:[5,6],
         3:[1],
         4:[1],
         5:[2],
         6:[2],
         }

simple_connection_info= defaultdict(list)
simple_connection_info.update(graph)


visted = bfs(simple_connection_info, 0)

visted = dfs(simple_connection_info, 0)
