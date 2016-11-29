# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:20:02 2016

@author: mic
"""

import pickle,pprint
import numpy as np
import random
#from sklearn.neural_network import BernoulliRBM
#import theano
#import theano.tensor as T

train_file_string = "train_forstu.pickle"
valid_file_string = "valid_forstu.pickle"
train_file = open(train_file_string, "rb")
valid_file = open(valid_file_string,"rb")

a = pickle.load(train_file,encoding = "latin1")
b = pickle.load(valid_file,encoding = "latin1")
train_file.close();
valid_file.close();
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

'''
#设置参数  
num_example = num_train
nn_input_dim=256 #输入神经元个数  
nn_output_dim=6 #输出神经元个数  
nn_hdim=100  
#梯度下降参数  
epsilon=0.01 #learning rate  
reg_lambda=0.01 #正则化长度 
w1=theano.shared(np.random.randn(nn_input_dim,nn_hdim),name="W1")  
b1=theano.shared(np.zeros(nn_hdim),name="b1")  
w2=theano.shared(np.random.randn(nn_hdim,nn_output_dim),name="W2")  
b2=theano.shared(np.zeros(nn_output_dim),name="b2")  

#前馈算法  
X=T.matrix('X')  #double类型的矩阵  
y=T.lvector('y') #int64类型的向量  
z1=X.dot(w1)+b1   #1  
a1=T.tanh(z1)     #2  
z2=a1.dot(w2)+b2  #3  
y_hat=T.nnet.softmax(z2) #4  
#正则化项  
loss_reg=1./num_example * reg_lambda/2 * (T.sum(T.square(w1))+T.sum(T.square(w2))) #5  
loss=T.nnet.categorical_crossentropy(y_hat,y).mean()+loss_reg  #6  
#预测结果  
prediction=T.argmax(y_hat,axis=1) #7  

forword_prop=theano.function([X],y_hat)  
calculate_loss=theano.function([X,y],loss)  
predict=theano.function([X],prediction)  

#求导  
dw2=T.grad(loss,w2)  
db2=T.grad(loss,b2)  
dw1=T.grad(loss,w1)  
db1=T.grad(loss,b1)  
  
#更新值  
gradient_step=theano.function(  
    [X,y],  
    updates=(  
        (w2,w2-epsilon*dw2),  
        (b2,b2-epsilon*db2),  
        (w1,w1-epsilon*dw1),  
        (b1,b1-epsilon*db1)  
  
    )  
)  
    
def build_model(num_passes=20000,print_loss=False):
    w1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))  
    b1.set_value(np.zeros(nn_hdim))  
    w2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))  
    b2.set_value(np.zeros(nn_output_dim))  
      
    for i in range(0,num_passes):  
        gradient_step(train_X,train_y)  
        if print_loss and i%1000==0:  
            print ("Loss after iteration %i: %f" ,(i,calculate_loss(train_X,train_y))) 
'''