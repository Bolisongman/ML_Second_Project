#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 18:55:19 2018

@author: alireza
"""

# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from surprise import model_selection

import time
import random
import numpy as np

import matplotlib.pyplot as plt

import utils
# %% Random Seed
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

# %% Load Project Dataset in a Surprise format
file_path = "Data/data_train_Surprise_format.csv"

reader = Reader(line_format='item user rating', sep=',',
                rating_scale=(1, 5), skip_lines=1)

data_train = Dataset.load_from_file(file_path, reader=reader)

# %% Hyper parameter tuning and CV analysis
# Algorithm: SVD
Hyper_Params = {'n_epochs': [50],
                'n_factors': [15, 25, 35],
                'lr_all': [0.005],
                'reg_all': [0.001, 0.01, 0.1, 0.2, 0.3]}

start = time.time()

Train_CV = Grid_Search_Result = model_selection.GridSearchCV(SVDpp,
                                                             Hyper_Params,
                                                             measures=['rmse'],
                                                             cv=3, n_jobs=3)

Train_CV.fit(data_train)

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Figures

plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_reg_all'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$)')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$)')
plt.savefig('3_fold_CV_Reg_Param.png')

plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_n_factors'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xlabel('Number of Factores')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Number of Factors')
plt.savefig('3_fold_CV_Factors.png')

# %% Best Hyper-parameters Training
alg = SVDpp()

alg.biased = Grid_Search_Result.best_params['rmse']['biased']
alg.n_epochs = Grid_Search_Result.best_params['rmse']['n_epochs']
alg.n_factors = Grid_Search_Result.best_params['rmse']['n_factors']
alg.reg_pu = Grid_Search_Result.best_params['rmse']['reg_all']
alg.reg_qi = Grid_Search_Result.best_params['rmse']['reg_all']
alg.reg_yj = Grid_Search_Result.best_params['rmse']['reg_all']
alg.reg_bi = Grid_Search_Result.best_params['rmse']['reg_all']
alg.reg_bu = Grid_Search_Result.best_params['rmse']['reg_all']
alg.lr_pu = Grid_Search_Result.best_params['rmse']['lr_all']
alg.lr_qi = Grid_Search_Result.best_params['rmse']['lr_all']

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
