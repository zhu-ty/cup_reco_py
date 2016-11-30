# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:26:35 2016

reference:
    http://blog.csdn.net/u014365862/article/details/48057053

@author: ShadowK
"""

import numpy as np  
import matplotlib.pylab as plt  
import copy  
from scipy.linalg import norm  
from math import pow  
from scipy.optimize import fminbound,minimize 
import pickle

def _dot(a, b):  
    # 表示点乘
    mat_dot = np.dot(a, b)  
    # exp表示e的多次次幂
    return np.exp(mat_dot)  

def condProb(theta, thetai, xi):  
    # 计算样本i的输出
    numerator = _dot(thetai, xi.transpose())  

    # 计算所有样本的输出求和
    denominator = _dot(theta, xi.transpose())  
    denominator = np.sum(denominator, axis=0)  
    p = numerator / denominator  
    return p 

'''  下面就是求得代价函数 '''
def costFunc(alfa, *args):  
    i = args[2]  
    original_thetai = args[0]  
    delta_thetai = args[1]  
    x = args[3]  
    y = args[4]  
    lamta = args[5]  

    labels = set(y)  
    thetai = original_thetai  
    thetai[i, :] = thetai[i, :] - alfa * delta_thetai  
    k = 0  
    sum_log_p = 0.0  
    for label in labels:  
        index = y == label  
        xi = x[index]  
        p = condProb(original_thetai,thetai[k, :], xi)  
        log_p = np.log10(p)  
        sum_log_p = sum_log_p + log_p.sum()  
        k = k + 1  
    # 这就是代价函数的公式
    r = -sum_log_p / x.shape[0]+ (lamta / 2.0) * pow(norm(thetai),2)  
     #print r ,alfa  

    return r

 

class skr:
    def __init__(self):
        self.alfa = 0  
        self.lamda = 0  
        self.feature_num = 0  
        self.label_num = 0  
        self.run_times = 0  
        self.col = 0  
        self.theta = [0]
        self.MAX_VAL = 0
    
    def init(self, feature_num, label_mum, run_times = 3000, col = 1e-6, alfa = 0.4, lamda = 0.0):
        self.alfa = alfa  
        self.lamda = lamda  
        self.feature_num = feature_num  
        self.label_num = label_mum  
        self.run_times = run_times  
        self.col = col  
        self.theta = np.random.random((label_mum, feature_num + 1)) + 1.0
        return
    def train(self, train_data, valid_data, model__):
        self.MAX_VAL = train_data[0].max()
        self.train_(train_data[0] / self.MAX_VAL,train_data[1].astype(int))
        x = self.predict_(valid_data[0] / self.MAX_VAL)
        suc = 0
        err = 0
        for i in range(valid_data[0].shape[0]):
            if(valid_data[1][i] == x[i, 0]):
                suc = suc + 1
            else:
                err = err + 1
        weight_file = open(model__,"wb")
        pickle.dump([self.theta, self.MAX_VAL], weight_file)
        weight_file.close()
        return suc / (suc + err)
    def predict(self, valid_data, model__):
        weight_file = open(model__,"rb")
        self.theta,self.MAX_VAL = pickle.load(weight_file)
        weight_file.close()
        return (self.predict_(valid_data[0] / self.MAX_VAL))[:, 0]
    def oneDimSearch(self, original_thetai,delta_thetai,i,x,y ,lamta):  
        res = minimize(costFunc, 0.0, method = 'Powell', args =(original_thetai,delta_thetai,i,x,y ,lamta))  
        return res.x  
    def train_(self, x, y):  
        tmp = np.ones((x.shape[0], x.shape[1] + 1))  
        tmp[:,1:tmp.shape[1]] = x  
        x = tmp  
        del tmp  
        labels = set(y)  
        self.errors = []  
        old_alfa = self.alfa  
        for kk in range(0, self.run_times):  
            i = 0  
              
            for label in labels:  
                tmp_theta = copy.deepcopy(self.theta)  
                one = np.zeros(x.shape[0])  
                index = y == label  
                one[index] = 1.0  
                thetai = np.array([self.theta[i, :]])  
                prob = self.condProb(thetai, x)  
                prob = np.array([one - prob])  
                prob = prob.transpose()  
                delta_thetai = - np.sum(x * prob, axis = 0)/ x.shape[0] + self.lamda * self.theta[i, :]  
                #alfa = self.oneDimSearch(self.theta,delta_thetai,i,x,y ,self.lamda)#一维搜索法寻找最优的学习率，没有实现  
                self.theta[i,:] = self.theta[i,:] - self.alfa * np.array([delta_thetai])  
                i = i + 1
            tmpp = self.performance(tmp_theta)
            self.errors.append(tmpp)
            if(kk % 100 == 0):
                print("iter : %d"%kk)
                print("error : %f"%tmpp)
          
    def performance(self, tmp_theta):  
        return norm(self.theta - tmp_theta)   
    def dot(self, a, b):  
        mat_dot = np.dot(a, b)  
        return np.exp(mat_dot)  
    def condProb(self, thetai, xi):  
        numerator = self.dot(thetai, xi.transpose())  
          
        denominator = self.dot(self.theta, xi.transpose())  
        denominator = np.sum(denominator, axis=0)  
        p = numerator[0] / denominator  
        return p  
    def predict_(self, x):  
        tmp = np.ones((x.shape[0], x.shape[1] + 1))  
        tmp[:,1:tmp.shape[1]] = x  
        x = tmp  
        row = x.shape[0]  
        col = self.theta.shape[0]  
        pre_res = np.zeros((row, col))  
        for i in range(0, row):  
            xi = x[i, :]  
            for j in range(0, col):  
                thetai = self.theta[j, :]  
                p = self.condProb(np.array([thetai]), np.array([xi]))  
                pre_res[i, j] = p  
        r = []  
        for i in range(0, row):  
            tmp = []  
            line = pre_res[i, :]  
            ind = line.argmax()  
            tmp.append(ind)  
            tmp.append(line[ind])  
            r.append(tmp)  
        return np.array(r)  
    def evaluate(self):  
        pass  