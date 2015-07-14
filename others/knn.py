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

train_data = pd.read_csv('../common/data/train_data_fe.csv')
test_data = pd.read_csv('../common/data/test_data_fe.csv')
valid_data = pd.read_csv('../common/data/valid_data_fe.csv')

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


'''
b = pd.read_csv('ft.csv',index_col=0)
b = np.array(b)

valid_data = valid_data * b.T
test_data = test_data * b.T
train_data = train_data * b.T
'''
#train_data = train_data[0:100,:]
#train_class = train_class[0:100]
svc = knc()
nb = range(31,52,5)
svm_parameters = {'n_neighbors':nb}

clf = gsc(svc, svm_parameters)

clf.fit(train_data,train_class)


print clf.grid_scores_
print clf.best_score_
print clf.best_estimator_
print clf.best_params_
