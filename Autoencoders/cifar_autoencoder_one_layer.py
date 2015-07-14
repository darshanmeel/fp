# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 13:26:22 2015

@author: Inpiron
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print parentdir
moduledir = parentdir + '/common'
print moduledir
sys.path.insert(0,moduledir) 
moduledir = parentdir + '/NeuralNet'
print moduledir
sys.path.insert(0,moduledir) 

from autoencoder_single_hidden_layer import autoencoder_single_hidden_layer
from NN_single_hidden_layer import NN_single_hidden_layer
import numpy as np
import pandas as pd

from active_functions import *
from loss_functions import *

train_data = pd.read_csv('../common/data/preprocessed/grey_data/train_data.csv')
test_data = pd.read_csv('../common/data/preprocessed/grey_data/test_data.csv')
valid_data = pd.read_csv('../common/data/preprocessed/grey_data/valid_data.csv')

train_class = pd.read_csv('../common/data/train_class.csv')
test_class = pd.read_csv('../common/data/test_class.csv')
valid_class = pd.read_csv('../common/data/valid_class.csv')

train_data = np.array(train_data)

test_data = np.array(test_data)
valid_data = np.array(valid_data)

train_class = np.ravel(np.array(train_class))
test_class = np.ravel(np.array(test_class))
valid_class = np.ravel(np.array(valid_class))

print train_data.shape
print test_data.shape
print valid_data.shape


n_input = train_data.shape[1]
n_hidden = 768
n_out = n_input
print n_out

lr = [0.01,0.01,0.1,0.5]
rg = [0.0001,0.001,0.1]
for learning_rate in lr:
    for reg in rg:
        nn1 = autoencoder_single_hidden_layer(n_hidden,n_input,n_out,epochs = 50,batchsize=5,learning_rate=learning_rate,loss_fnc=rms_reg,reg = reg,momentum = 0.95)
        train_error,valid_error = nn1.fit(train_data,test_data)
            
        print train_error
        print
        print valid_error

print ghyu
_,train_data = nn1.predict(train_data)
_,test_data = nn1.predict(test_data)

train_data = np.array(train_data)
test_data = np.array(test_data)


n_input = train_data.shape[1]
n_hidden = 256
n_out = n_input
print n_out
nn1 = autoencoder_single_hidden_layer(n_hidden,n_input,n_out,epochs = 50,batchsize=5,learning_rate=0.01,loss_fnc=rms_reg,reg = 0.0001,momentum = 0.95)
train_error,valid_error = nn1.fit(train_data,test_data)
    
print train_error
print
print valid_error

_,train_data = nn1.predict(train_data)
_,test_data = nn1.predict(test_data)

train_data = np.array(train_data)
test_data = np.array(test_data)



n_input = train_data.shape[1]
n_hidden = 128
n_out = n_input
print n_out
nn1 = autoencoder_single_hidden_layer(n_hidden,n_input,n_out,epochs = 80,batchsize=5,learning_rate=0.01,loss_fnc=rms_reg,reg = 0.0001,momentum = 0.95)
train_error,valid_error = nn1.fit(train_data,test_data)
    
print train_error
print
print valid_error

_,train_data = nn1.predict(train_data)
_,test_data = nn1.predict(test_data)

train_data = np.array(train_data)
test_data = np.array(test_data)

    

n_input = train_data.shape[1]
n_out = len(np.unique(train_class))
print n_out
for n_hidden in range(24,121,8):
    print
    print n_hidden
    nn1 = NN_single_hidden_layer(n_hidden,n_input,n_out,epochs = 50,batchsize=5,learning_rate=0.1,loss_fnc=rms_cls,reg = 0.00001,momentum = 0.95)
    train_error,valid_error = nn1.fit(train_data,train_class,test_data,test_class)
        
    print train_error
    print
    print valid_error

'''
print
print nn1.nn['w1']
print
print nn1.nn['b1']
print
print nn1.nn['w2']
print
print nn1.nn['b2']
'''