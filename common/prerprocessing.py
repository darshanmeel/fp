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

from sklearn.preprocessing import MinMaxScaler 
IMGSIZE = 32
DEPTH = 3

#remove the mean from each image. In natural images it might not be needed.But it doesnt harm.
def remove_mean_from_image(X):
    X = np.reshape(X,(X.shape[0],IMGSIZE*IMGSIZE,DEPTH))
    mns = np.mean(X,axis=1)
    mns = np.reshape(mns,(mns.shape[0],1,mns.shape[1]))
    X = X-mns
    return(X)
    
# min_max scalar
def min_max_standarization(X):
    X = np.reshape(X,(X.shape[0],-1))
    mms = MinMaxScaler(copy=False)
    mms.fit_transform(X)
    return(X)
    
def find_centers(X):
    X = np.reshape(X,(X.shape[0],IMGSIZE*IMGSIZE,DEPTH))
    cntrs = np.mean(X,axis=0)
    return(cntrs)
    
    
train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')
valid_data = pd.read_csv('data/valid_data.csv')

train_class = pd.read_csv('data/train_class.csv')
train_class = np.ravel(np.array(train_class))



train_data = np.array(train_data)

test_data = np.array(test_data)
valid_data = np.array(valid_data)

train_data = remove_mean_from_image(train_data)
test_data = remove_mean_from_image(test_data)
valid_data = remove_mean_from_image(valid_data)

print train_data.shape
print train_data[0,:,:]

train_data = min_max_standarization(train_data)
test_data = min_max_standarization(test_data)
valid_data = min_max_standarization(valid_data)

print train_data.shape
print train_data[0,:]


overallcntrs = find_centers(train_data)
oc = pd.DataFrame(overallcntrs)
oc.to_csv('data/oc.csv',index=False)

#find the centers for points in each class
cls_cntrs = []
for cls in range(10):
    td = train_data[np.where(train_class == cls)]
    cc= find_centers(td)

    cls_cntrs.append(find_centers(td))

cc = np.array(cls_cntrs)
cc = np.reshape(cc,(cc.shape[0],-1))
cc = pd.DataFrame(cc)
print cc
cc.to_csv('data/cc.csv',index=False)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
valid_data = pd.DataFrame(valid_data)

train_data.to_csv('data/train_data_p.csv',index=False)
test_data.to_csv('data/test_data_p.csv',index=False)
valid_data.to_csv('data/valid_data_p.csv',index=False)
