# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:24:57 2019

@author: comingboy
"""

import tarfile
import os
def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs) 

if __name__ == "__main__":
    untar("gensim-3.8.0.tar.gz", ".")