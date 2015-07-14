# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 15:27:31 2015

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


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.grid_search import GridSearchCV as gsc
from sklearn.preprocessing import MinMaxScaler 

# min_max scalar
def min_max_standarization(X):
    X = np.reshape(X,(X.shape[0],-1))
    mms = MinMaxScaler(copy=False)
    mms.fit_transform(X)
    return(X)
    
    
IMGSIZE = 32
DEPTH = 3
BUCKETSIZE = 16

def fr(a):


    frs = np.zeros(BUCKETSIZE)
    vals = np.unique(a,return_counts=True)

    rw =list(vals[0])

    cl = vals[1]

    for i,val in enumerate(rw):
        frs[val] = cl[i]

    return (frs)

def dist_from_centers(X,cntr):
    X = np.reshape(X,(X.shape[0],IMGSIZE*IMGSIZE,DEPTH))
    cntr = np.reshape(cntr,(IMGSIZE*IMGSIZE,DEPTH))
    dist = np.sqrt(np.sum(np.square(X-cntr),axis=1))
    return(dist)


train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')
valid_data = pd.read_csv('data/valid_data.csv')


train_data = np.array(train_data)
test_data = np.array(test_data)
valid_data = np.array(valid_data)


train_class = pd.read_csv('data/train_class.csv')
test_class = pd.read_csv('data/test_class.csv')
valid_class = pd.read_csv('data/valid_class.csv')


train_class = np.ravel(np.array(train_class))
test_class = np.ravel(np.array(test_class))
valid_class = np.ravel(np.array(valid_class))

#find color histograms
train_data_chst =  np.array(train_data/BUCKETSIZE,dtype='int64').reshape(train_data.shape[0],IMGSIZE*IMGSIZE,DEPTH)
train_extra_features = np.apply_along_axis(fr,1,train_data_chst).reshape(train_data_chst.shape[0],-1)

test_data_chst =  np.array(test_data/BUCKETSIZE,dtype='int64').reshape(test_data.shape[0],IMGSIZE*IMGSIZE,DEPTH)
test_extra_features = np.apply_along_axis(fr,1,test_data_chst).reshape(test_data_chst.shape[0],-1)

valid_data_chst =  np.array(valid_data/BUCKETSIZE,dtype='int64').reshape(valid_data.shape[0],IMGSIZE*IMGSIZE,DEPTH)
valid_extra_features = np.apply_along_axis(fr,1,valid_data_chst).reshape(valid_data_chst.shape[0],-1)

#now get the preprocessed data

train_data = pd.read_csv('data/train_data_p.csv')
test_data = pd.read_csv('data/test_data_p.csv')
valid_data = pd.read_csv('data/valid_data_p.csv')


train_data = np.array(train_data)
test_data = np.array(test_data)
valid_data = np.array(valid_data)

oc = pd.read_csv('data/oc.csv')
oc = np.array(oc)

dist = dist_from_centers(train_data,oc)
train_extra_features = np.hstack((train_extra_features,dist))

dist = dist_from_centers(test_data,oc)
test_extra_features = np.hstack((test_extra_features,dist))

dist = dist_from_centers(valid_data,oc)
valid_extra_features = np.hstack((valid_extra_features,dist))

cc = pd.read_csv('data/cc.csv')
cc = np.array(cc)

for i in range(10):
    cntr = cc[i,:]
    dist = dist_from_centers(train_data,cntr)
    train_extra_features = np.hstack((train_extra_features,dist))
    
    dist = dist_from_centers(test_data,oc)
    test_extra_features = np.hstack((test_extra_features,dist))

    dist = dist_from_centers(valid_data,oc)
    valid_extra_features = np.hstack((valid_extra_features,dist))

print train_extra_features.shape
print test_extra_features.shape
print valid_extra_features.shape

# use convolution to get more features. Use a filter of size 5 by 5 followed by pooling

train_extra_features = min_max_standarization(train_extra_features)
test_extra_features = min_max_standarization(test_extra_features)
valid_extra_features = min_max_standarization(valid_extra_features)

print train_extra_features.shape
print test_extra_features.shape
print valid_extra_features.shape


train_data = np.hstack((train_data,train_extra_features))
test_data = np.hstack((test_data,test_extra_features))
valid_data = np.hstack((valid_data,valid_extra_features))


train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
valid_data = pd.DataFrame(valid_data)

train_data.to_csv('data/train_data_fe.csv',index=False)
test_data.to_csv('data/test_data_fe.csv',index=False)
valid_data.to_csv('data/valid_data_fe.csv',index=False)


 

