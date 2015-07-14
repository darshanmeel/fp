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


train_data = pd.read_csv('../common/data/train_data_p.csv')
test_data = pd.read_csv('../common/data/test_data_p.csv')
valid_data = pd.read_csv('../common/data/valid_data_p.csv')

train_class = pd.read_csv('../common/data/train_class.csv')
test_class = pd.read_csv('../common/data/test_class.csv')
valid_class = pd.read_csv('../common/data/valid_class.csv')

train_data = np.array(train_data)
test_data = np.array(test_data)
valid_data = np.array(valid_data)

train_class = np.ravel(np.array(train_class))
test_class = np.ravel(np.array(test_class))
valid_class = np.ravel(np.array(valid_class))


'''
cols = pd.read_csv('C:\Users\Inpiron\Documents\col.csv')
cols = np.array(cols)
cols = list(cols[:,0])
print cols


train_data = train_data[:,cols]
test_data = test_data[:,cols]
valid_data = valid_data[:,cols] 

print train_data.shape
print test_data.shape
print valid_data.shape
'''
#train_class = train_class[0:100]

svc = rfc(n_estimators=500, min_samples_split = 9,criterion='gini')
svm_parameters = {'n_estimators':[500], 'min_samples_split' : [9]}

clf = gsc(svc, svm_parameters)
clf = svc

clf.fit(train_data,train_class)
print

print clf.score(valid_data,valid_class)
print clf.score(test_data,test_class)
#print svc.feature_importances_
'''
print clf.grid_scores_

print clf.best_score_
print clf.best_estimator_
print clf.best_params_
'''