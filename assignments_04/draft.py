# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:24:14 2019

@author: ASUSNB
"""

input_file = 'zh_wiki_00'
n = 0
outfile= open('zh_wiki_01',encoding='utf-8', mode = 'w')
with open(input_file, encoding='utf-8', mode = 'r') as f:
    for line in f:
        outfile.write(line+'\n')
        print(line)
        n +=1
        if n>=200000:
            break
outfile.close()