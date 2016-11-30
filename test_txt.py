# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 23:13:22 2016

@author: ShadowK
"""
import pickle
import numpy as np

valid__ = "data/valid_forstu.pickle"
txt__ = "output/predict_bp.txt"

valid_ = open(valid__,"rb")
txt_ = open(txt__,"r")

a,b = pickle.load(valid_, encoding = "latin1")

cor = 0;
err = 0;
for i in range(0, b.shape[0]):
    x = float(txt_.readline())
    if(x == b[i]):
        cor = cor + 1
    else:
        err = err + 1

print(cor / (cor + err))
