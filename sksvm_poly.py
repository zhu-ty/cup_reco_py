# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:52:17 2016

@author: ShadowK
"""
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np

class skr:
    def init(self, input_num, output_num):
        return
    def train(self, train_data, valid_data, model__):
        clf_linear = SVC(kernel = "poly",degree = 3)
        clf_linear.fit(train_data[0], train_data[1]) 
        acc_linear = clf_linear.score(valid_data[0],valid_data[1])
        
        joblib.dump(clf_linear,model__)
        
        return acc_linear
    def predict(self, valid_data, model__):
        clf = joblib.load(model__)
        return clf.predict(valid_data[0])