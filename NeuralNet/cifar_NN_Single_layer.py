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

from active_functions import *
from loss_functions import *
from NN_single_hidden_layer import NN_single_hidden_layer
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split


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

cols = pd.read_csv('C:\Users\Inpiron\Documents\col.csv')
cols = np.array(cols)
cols = list(cols[:,0])
print cols

'''
train_data = train_data[:,cols]
test_data = test_data[:,cols]
valid_data = valid_data[:,cols] 
'''
print train_data.shape
print test_data.shape
print valid_data.shape

n_input = train_data.shape[1]
n_out = len(np.unique(train_class))
print n_out
for n_hidden in range(572,573,8):
    print
    print 'starts'
    print n_hidden
    nn1 = NN_single_hidden_layer(n_hidden,n_input,n_out,epochs = 200,batchsize=5,learning_rate=0.1,loss_fnc=rms_cls,reg = 0.00001,momentum = 0.95)
    train_error,valid_error = nn1.fit(train_data,train_class,test_data,test_class)
        
    print train_error
    print
    print valid_error
    print
    print 'ends'
    print
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