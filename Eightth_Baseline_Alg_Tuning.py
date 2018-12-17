#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:43:21 2018

@author: alireza
"""

# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import model_selection

import time
import random
import numpy as np

import matplotlib.pyplot as plt

import utils


# %% Random Seed
my_seed = 10
random.seed(my_seed)
np.random.seed(my_seed)


# %% Load Project Dataset in a Surprise format
file_path = "Data/data_train_Surprise_format.csv"

reader = Reader(line_format='item user rating', sep=',',
                rating_scale=(1, 5), skip_lines=1)

data_train = Dataset.load_from_file(file_path, reader=reader)

# %% Hyper parameter tuning and CV analysis
Hyper_Params = {'bsl_options':
                {'method': ['als'],
                'n_epochs': [20],
                'reg_u': [0.001, 0.01, 0.1],
                'reg_i': [0.001, 0.01, 0.1]}}

start = time.time()
Train_CV = Grid_Search_Result = model_selection.GridSearchCV(BaselineOnly,
                                                             Hyper_Params,
                                                             measures=['rmse'],
                                                             cv=3, n_jobs=3,
                                                             return_train_measures=True,
                                                             joblib_verbose=3)

Train_CV.fit(data_train)

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)


# %% Figures
reg_i = []
reg_u = []
for i in Train_CV.cv_results['param_bsl_options']:
    reg_i.append(i['reg_i'])
    reg_u.append(i['reg_u'])

reg_i = np.array(reg_i)
reg_u = np.array(reg_u)

plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(reg_i,
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$) - bi')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$) - bi')
plt.savefig('3_fold_CV_Reg_Param_Baseline_bi.png')

plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(reg_u,
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$) - bu')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$) - bu')
plt.savefig('3_fold_CV_Reg_Param_Baseline_bu.png')

# %% Best Hyper-parameters Training
alg = BaselineOnly

alg.bsl_options = Grid_Search_Result.best_params['rmse']['bsl_options']

start = time.time()

alg.fit(data_train.build_full_trainset())

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Loading Test Data
file_path = "Data/sample_submission.csv"
data_test = utils.load_data_desired(file_path)

# %% Prediction
Predict_Test = []

for line in data_test:
    Predict_Test.append(alg.predict(str(line[1]), str(line[0])).est)


# %% Save Prediction
file = open("Details.txt", "w")

file.write("+ Best Score: \n \n")
file.write(str(Train_CV.best_score) + "\n \n")
file.write("************************************************************ \n")
file.write("+ Best Param: \n \n")
file.write(str(Train_CV.best_params) + "\n \n")
file.write("************************************************************ \n")
file.write("+ CV Summary: \n \n")
file.write(str(Train_CV.cv_results) + "\n \n")
file.write("************************************************************ \n")

file.close()

# %% Save Prediction
file = open("testfile.csv", "w")
file.write("Id,Prediction\n")

for i in range(len(Predict_Test)):
    line = data_test[i]
    temp = 'r' + str(line[0]) + '_c' + str(line[1]) + ',' + str(int(round(Predict_Test[i]))) + '\n'
    file.write(temp)

file.close()