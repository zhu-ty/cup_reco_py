# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:57:16 2016

@author: ShadowK
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

class skr:
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
        self.iters_ = 0
        self.learn_ = 0
        self.correct_ = 0
        
    def init(self, input_num, output_num, hidden = 100, iters = 10,learn = 0.05, correct = 0.1):
        self.setup(input_num, hidden, output_num)
        self.iters_ = iters
        self.learn_ = learn
        self.correct_ = correct
        
    def train(self, train_data, valid_data, model__):
        #print("start")
        num_samples = train_data[0].shape[0]
        a11 = np.zeros([num_samples, self.output_n])
        for i in range(0, num_samples):
            a11[i, int(train_data[1][i])] = 1
        print("start train iter %d"%self.iters_)
        self.train_(train_data[0], a11, self.iters_, self.learn_, self.correct_)
        i = 0;
        x = np.zeros([valid_data[0].shape[0], self.output_n])
        y = np.zeros([valid_data[0].shape[0], 1])
        cor = 0
        err = 0
        for case in valid_data[0]:
            #print(self.predict(case))
            x[i,:] = self.predict_(case)
            y[i] = (np.where(x[i,:] == x[i,:].max()))[0][0]
            if(y[i] == valid_data[1][i]):
                cor = cor + 1
            else:
                err = err + 1
            i = i + 1
        weight_file = open(model__,"wb")
        pickle.dump([self.input_weights,self.output_weights],weight_file)
        weight_file.close()
        return (cor/(cor+err))
        
    def predict(self, valid_data, model__):
        model_ = open(model__,"rb")
        self.input_weights,self.output_weights = pickle.load(model_, encoding = "latin1")
        model_.close()
        i = 0;
        x = np.zeros([valid_data[0].shape[0], self.output_n])
        y = np.zeros([valid_data[0].shape[0]])
        for case in valid_data[0]:
            #print(self.predict(case))
            x[i,:] = self.predict_(case)
            y[i] = (np.where(x[i,:] == x[i,:].max()))[0][0]
            i = i + 1
        return y

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

    def predict_(self, inputs):
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
        self.predict_(case)
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

    def train_(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            print("iter : %d"%j)
            print("error : %f"%error)