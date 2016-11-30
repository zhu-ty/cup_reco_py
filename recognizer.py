# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:55:45 2016

@author: ShadowK
"""
#print("Hello World!") 

import numpy as np
import pprint
import sys,getopt
import pickle,random
random.seed(0)

import sksvm_linear as sksl
import sksvm_poly as sksp
import skbp
import sksoftmax as sksm

def usage():
    print("Useage:")
    print("python recognizer.py -t [--svml | --svmp | --skbp | --sksm] [--ti data/train_forstu.pickle] [--tv data/valid_forstu.pickle] [--to model/model.model]")
    print("python recognizer.py -p [--svml | --svmp | --skbp | --sksm] [-i data/valid_forstu.pickle] [-o output/predict.txt] [-m model/model.model]")
    print("Sample:")
    print("python recognizer.py -t --sksm --ti data/train_forstu.pickle --tv data/valid_forstu.pickle --to model/softmax.model")
    print("python recognizer.py -p --bpsk -i data/valid_forstu.pickle -o output/predict.txt -m model/bp.model")
    return


   
    
if(sys.argv[1:] == []):
    usage()
    sys.exit(0)
    
opts, args = getopt.getopt(sys.argv[1:], "tpi:o:m:",["svml","svmp","sksm","skbp","bp_iter=","ti=","to=","tv="])

mode = 0
method = 0
bp_iter = 10

train_dataset__ = ""
valid_dataset__ = ""

model__ = ""

data__ = ""
output__ = ""



for op, value in opts:
    if op == "-t":
        mode = 1
    if op == "-p":
        mode = 2
    if(mode == 0):
        print("Please first tell me if you want to train or predict")
        usage()
        sys.exit(0)
    if op == "--svml":
        method = 1
    elif op == "--svmp":
        method = 2
    elif op == "--skbp":
        method = 3
    elif op == "--sksm":
        method = 4
    elif op == "--ti":
        train_dataset__ = value
    elif op == "--tv":
        valid_dataset__ = value
    elif op == "-i":
        data__ = value
    elif op == "-o":
        output__ = value
    elif op == "--bp_iter":
        bp_iter = int(value)
    if(mode == 1):
        if op == "--to":
            model__ = value
    elif(mode == 2):
        if op == "-m":
            model__ = value


'''
skr()
init(self, input_num, output_num))
acc = train(train_data, valid_data, model__)
ans = predict(valid_data, model__)


'''
            
if method == 1:
    skr = sksl.skr()
    skr.init(256, 6)
elif method == 2:
    skr = sksp.skr()
    skr.init(256, 6)
elif method == 3:
    skr = skbp.skr()
    skr.init(256, 6, 100, bp_iter)
elif method == 4:
    skr = sksm.skr()
    skr.init(256, 6)
#print("test1")    

if mode == 1:
    train_data_ = open(train_dataset__,"rb")
    train_data = pickle.load(train_data_,encoding = "latin1")
    train_data_.close()
    train_num = train_data[0].shape[0]
    index_train = np.array(range(train_num))
    random.shuffle(index_train)
    train_data[0] = train_data[0][index_train,:]
    train_data[1] = train_data[1][index_train]
    
    valid_data_ = open(valid_dataset__,"rb")
    valid_data = pickle.load(valid_data_,encoding = "latin1")
    valid_data_.close()
    #print("test2")
    suc_rate = skr.train(train_data, valid_data, model__)
    print("accuracy : %f"%suc_rate)
elif mode == 2:
    valid_data_ = open(data__,"rb")
    valid_data = pickle.load(valid_data_,encoding = "latin1")
    valid_data_.close()
    
    predicted_data = skr.predict(valid_data, model__)
    op_file = open(output__,"w")
    for i in range(0,predicted_data.shape[0]):
        op_file.write(str(predicted_data[i]))
        op_file.write("\n")
    op_file.close()