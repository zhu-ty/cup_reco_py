# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:55:05 2016

@author: mic
"""

import random
import math
import pickle
import numpy as np
random.seed(0)

def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):  # 创造一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat
    
    
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmod_derivate(x):
    return x * (1 - x)

class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmod_derivate(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for i in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            print(str(error))

    def test(self):
        train_file_string = "train_forstu.pickle"
        #valid_file_string = "valid_forstu.pickle"
        train_file = open(train_file_string, "rb")
        #valid_file = open(valid_file_string,"rb")
        
        a = pickle.load(train_file,encoding = "latin1")
        #b = pickle.load(valid_file,encoding = "latin1")
        train_file.close();
        #valid_file.close();
        a0 = a[0]
        a1 = a[1]
        num_train = a0.shape[0]
        index_train = np.array(range(num_train))
        random.shuffle(index_train)
        a0 = a0[index_train,:]
        a1 = a1[index_train]
        
        a11 = np.zeros([num_train,6])
        
        for i in range(0,num_train):
            a11[i,int(a1[i])] = 1
        
        train_X = a0;
        train_y = a11;
        self.setup(256,100,6)
        self.train(train_X, train_y, 20, 0.05, 0.1)
        #for case in cases:
            #print(self.predict(case))
            
    def mine(self,str_wei):
        weight_file_string = str_wei
        weight_file = open(weight_file_string,"rb")
        weight = pickle.load(weight_file,encoding = "latin1")
        weight_file.close()
        
        self.setup(256,100,6)
        
        self.input_weights = weight[0]
        self.output_weights = weight[1]
        
        valid_file_string = "valid_forstu.pickle"
        valid_file = open(valid_file_string,"rb")
        b = pickle.load(valid_file,encoding = "latin1")
        valid_file.close()
        i = 0;
        x = np.zeros([b[0].shape[0],6])
        y = np.zeros([b[0].shape[0],1])
        cor = 0
        err = 0
        for case in b[0]:
            #print(self.predict(case))
            x[i,:] = self.predict(case)
            y[i] = (np.where(x[i,:] == x[i,:].max()))[0][0]
            #print(i)
            if(y[i] == b[1][i]):
                cor = cor + 1
            else:
                err = err + 1
            i = i + 1
        print(cor/(cor+err))
        return y
            
    def save(self,str_wei):
        weight_file = open(str_wei,"wb")
        pickle.dump([self.input_weights,self.output_weights],weight_file)
        weight_file.close()


nn = BPNeuralNetwork()
nn.test()
nn.save("weights2.pickle")
xy = nn.mine("weights2.pickle")
    