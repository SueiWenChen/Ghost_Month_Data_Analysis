#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:15:36 2021

@author: zqelaine
"""
#%% import packages and load data
import os
import numpy as np 
import pandas as pd
import scipy
from scipy import stats

os.chdir('/Users/zqelaine/Desktop/IDS/FINAL project')

# load the dataset movieReplicationSet
data = pd.read_csv('DrowningData.csv', encoding='utf-8')


#%% Question 2: What reason(s) for drowning are more prevalent in the ghost month than other months? Do they match?

# split the dataset into two groups: ghost month, other months
ghost_month = data[data['CC_Month']==7]
other_months = data[data['CC_Month']!=7]

# independent t-test
reasons = ("Work","Suicide","Floating Corpse","Capsizing","Slipping","Traffic Accident","Playing","Snorkeling","Saving Others","Fishing","Other")
numReasons = len(reasons)
t1 = np.empty([numReasons,1])
t1[:] = np.NaN
p1 = np.empty([numReasons,1])
p1[:] = np.NAN

for i in range(numReasons):
         
        # Extract the column of drowning reasons and the combine the two groups together into a dataset
        ghost_reason = np.array(np.where(ghost_month['Drowning_reasons'] == reasons[i], 1, 0))
        others_reason = np.array(np.where(other_months['Drowning_reasons'] == reasons[i], 1, 0))
        combinedData = np.transpose(np.array([ghost_reason,others_reason])) # array of arrays
        
        # independent t test
        t1[i],p1[i] = stats.ttest_ind(combinedData[0],combinedData[1],alternative='greater', equal_var=False)

