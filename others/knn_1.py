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

from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.grid_search import GridSearchCV as gsc


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


b = pd.read_csv('ft.csv',index_col=0)
b = np.array(b)

valid_data = valid_data * b.T
test_data = test_data * b.T
train_data = train_data * b.T
k = 101
nbs = 92
c = np.argsort(b,axis = 0)
for k in range(1,k,1):
    d = c[-k:-1,0]
    print d
    d = list(d)

    d.append(c[-1,0])

    td = train_data[:,d]
    tsd = test_data[:,d]
    vld = valid_data[:,d]
    #train_data = train_data[0:100,:]
    #train_class = train_class[0:100]
    for nb in range(91,nbs,1):    
        clf = knc(n_neighbors = nb)
        clf.fit(td,train_class)
        print
        print k,nb
        print 'scr'
        print clf.score(tsd,test_class)
    
    