#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:17:12 2018

@author: alireza
"""


# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise import SlopeOne
from surprise import SVD
from surprise import BaselineOnly
from surprise import KNNBasic

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

# %% Best Hyper-parameters Training - NMF
alg_NMF = NMF()

alg_NMF.biased = False
alg_NMF.n_epochs = 50
alg_NMF.n_factors = 35
alg_NMF.reg_pu = 0.1
alg_NMF.reg_qi = 0.1
alg_NMF.verbose = True

start = time.time()

alg_NMF.fit(data_train.build_full_trainset())

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Best Hyper-parameters Training - SVD
alg_SVD = SVD()

alg_SVD.biased = True
alg_SVD.n_epochs = 50
alg_SVD.n_factors = 35
alg_SVD.reg_pu = 0.1
alg_SVD.reg_qi = 0.1
alg_SVD.verbose = True

start = time.time()

alg_SVD.fit(data_train.build_full_trainset())

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Best Hyper-parameters Training - Slope One
alg_SL1 = SlopeOne()

start = time.time()

alg_SL1.fit(data_train.build_full_trainset())

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Best Hyper-parameters Training - KNN
sim_options = {'name': 'msd',
               'user_based': True  # compute  similarities between users
               }
alg_KNN = KNNBasic(sim_options=sim_options)

start = time.time()

alg_KNN.fit(data_train.build_full_trainset())

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Best Hyper-parameters Training - Base Line
bsl_options = {'method': 'als',
               'n_epochs': 20,
               'reg_u': 1,
               'reg_i': 10}

alg_BSL = BaselineOnly(bsl_options=bsl_options)

start = time.time()

alg_BSL.fit(data_train.build_full_trainset())

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Loading Test Data
file_path = "Data/sample_submission.csv"
data_test = utils.load_data_desired(file_path)

# %% Test Prediction
Pred_Test_SVD = []
Pred_Test_NMF = []
Pred_Test_SL1 = []
Pred_Test_KNN = []
Pred_Test_BSL = []

start = time.time()
for line in data_test:
    Pred_Test_KNN.append(alg_KNN.predict(str(line[1]),
                                         str(line[0]), clip=False).est)

    Pred_Test_SVD.append(alg_SVD.predict(str(line[1]),
                                         str(line[0]), clip=False).est)

    Pred_Test_NMF.append(alg_NMF.predict(str(line[1]),
                                         str(line[0]), clip=False).est)

    Pred_Test_SL1.append(alg_SL1.predict(str(line[1]),
                                         str(line[0]), clip=False).est)

    Pred_Test_BSL.append(alg_BSL.predict(str(line[1]),
                                         str(line[0]), clip=False).est)

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

X_Test = np.matrix([Pred_Test_SVD,
                    Pred_Test_NMF,
                    Pred_Test_SL1,
                    Pred_Test_KNN,
                    Pred_Test_BSL])
X_Test = X_Test.T

# %% Prior Based
X_Test = np.matrix([Pred_Test_SVD,
                    Pred_Test_NMF,
                    Pred_Test_SL1,
                    Pred_Test_KNN])
X_Test = X_Test.T

Pred_Test = np.mean(X_Test,axis=1)
Pred_Test = Pred_Test.A1
Pred_Test = utils.Prior_Correction(Pred_Test)

# #%% Save Prediction
file = open("SVD_BSL_NMF_SL1.csv", "w")
file.write("Id,Prediction\n")

for i in range(len(Pred_Test)):
    line = data_test[i]
    temp = 'r' + str(line[0]) + '_c' + str(line[1])\
        + ',' + str(int(Pred_Test[i])) + '\n'
    file.write(temp)

file.close()






