#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 19:38:46 2018

@author: alireza
"""

# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import NMF

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
file_path = "Data/data_train_preprocessed_surprise_format.csv"

reader = Reader(line_format='item user rating', sep=',',
                rating_scale=(1, 5), skip_lines=1)

data_train = Dataset.load_from_file(file_path, reader=reader)

# %% Best Hyper-parameters Training
alg = NMF()

alg.biased = False
alg.n_epochs = 50
alg.n_factors = 35
alg.reg_pu = 0.1
alg.reg_qi = 0.1
alg.verbose = True

start = time.time()

alg.fit(data_train.build_full_trainset())

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)

# %% Loading train data
file_path = "Data/data_train.csv"
data_train = utils.load_data_desired(file_path)

# %% Overall Labels for training
Pred_NotCliped_label = []
Real_label = []

Clip = False

for line in data_train:
    Real_label.append(line[2])
    Pred_NotCliped_label.append(alg.predict(str(line[1]),
                                            str(line[0]), clip=False).est)

Pred_NotCliped_label = np.array(Pred_NotCliped_label)
Real_label = np.array(Real_label)

# %% Prior CV
K = 3
K_Ind = utils.build_k_indices(y=Real_label, k_fold=K, seed=10)

Alpha = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])/10

RMSE = np.zeros((K, len(Alpha)))

Prior0 = np.ones(5)/5

for i in range(len(Alpha)):
    a = Alpha[i]
    for k in range(K):
        test_ind = []
        train_ind = []
        for j in range(K_Ind.shape[0]):
            if (j == k):
                test_ind.extend(K_Ind[j])
            else:
                train_ind.extend(K_Ind[j])

        x_test = Pred_NotCliped_label[test_ind]
        x_train = Pred_NotCliped_label[train_ind]

        y_test = Real_label[test_ind]
        y_train = Real_label[train_ind]

        Prior = np.histogram(x_train, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                             density=True)[0]
        Prior = ((1 - a) * Prior) + (a * Prior0)

        Noise_Var = np.mean(np.square(x_train - y_train))

        y_hat = utils.Prior_Correction(x_test, Noise_Var, Prior)

        RMSE[k, i] = np.sqrt(np.mean(np.square(y_test - y_hat)))


# %% Plot
meanRMSE = np.mean(RMSE, axis=0)

plt.figure()
plt.plot(Alpha, meanRMSE)
plt.grid()





