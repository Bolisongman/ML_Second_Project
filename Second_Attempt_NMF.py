#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:11:59 2018

@author: alireza
"""

# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise import model_selection

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

Hyper_Params = {'n_epochs': [50],
                'n_factors': [25],
                'biased': [True],
                'reg_pu': [0.1],
                'reg_qi': [0.1]}

Train_CV = Grid_Search_Result = model_selection.GridSearchCV(NMF,
                                                             Hyper_Params,
                                                             measures=['rmse'],
                                                             cv=3, n_jobs=3)

Train_CV.fit(data_train)

# %% Figures

plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_reg_qi'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$) - Qi')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$) - Qi')
plt.savefig('3_fold_CV_Reg_Param_NMF_Qi.png')

plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_reg_pu'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$) - Pu')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$) - Pu')
plt.savefig('3_fold_CV_Reg_Param_NMF_Pu.png')

plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_n_factors'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xlabel('Number of Factors')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Number of Factors')
plt.savefig('3_fold_CV_Reg_Param_NMF_n_factors.png')

# %% Best Hyper-parameters Training
alg = NMF()

alg.biased = Grid_Search_Result.best_params['rmse']['biased']
alg.n_epochs = Grid_Search_Result.best_params['rmse']['n_epochs']
alg.n_factors = Grid_Search_Result.best_params['rmse']['n_factors']
alg.reg_pu = Grid_Search_Result.best_params['rmse']['reg_pu']
alg.reg_qi = Grid_Search_Result.best_params['rmse']['reg_qi']

alg.fit(data_train.build_full_trainset())

# %% Loading Test Data
file_path = "Data/sample_submission.csv"
data_test = utils.load_data_desired(file_path)

# %% Prediction
Predict_Test = []

for line in data_test:
    Predict_Test.append(alg.predict(str(line[1]),str(line[0])).est)


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



