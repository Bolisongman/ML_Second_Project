#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:54:54 2018

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

# %% Look at the prior on the train data
file_path = "Data/data_train_preprocessed.csv"
data_train = utils.load_data_desired(file_path)

# %% Labels for training
Pred_NotCliped_label = []
Pred_Cliped_label = []
Real_label = []

Clip = False

for line in data_train:
    Real_label.append(line[2])
    Pred_NotCliped_label.append(alg.predict(str(line[1]),
                                            str(line[0]), clip=False).est)
    Pred_Cliped_label.append(alg.predict(str(line[1]),
                                         str(line[0]), clip=True).est)

Pred_NotCliped_label = np.array(Pred_NotCliped_label)
Pred_Cliped_label = np.array(Pred_Cliped_label)
Real_label = np.array(Real_label)

# %% Visualization
plt.figure()
plt.hist(Pred_NotCliped_label)
plt.grid()
plt.title('Histogram of Predicted Labels')
plt.xlabel('Label')

plt.figure()
plt.hist(np.round(Pred_Cliped_label),
         bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.grid()
plt.title('Histogram of Rounded Predicted Labels')
plt.xlabel('Label')
plt.xlim((0.5, 5.5))

plt.figure()
plt.hist(Real_label,
         bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.grid()
plt.title('Histogram of Real Labels')
plt.xlabel('Label')
plt.xlim((0.5, 5.5))

plt.figure()
plt.hist(Pred_NotCliped_label - Real_label)
plt.grid()
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')

# %% Dist Analysis
Priors = np.histogram(Real_label, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                      density=True)[0]
Sigma_Resid = np.mean(np.square(Pred_NotCliped_label - Real_label))

# %% New Prediction
Prior_Based_Prediction = \
    utils.Prior_Correction(Pred_NotCliped_label,
                           Noise_Var=Sigma_Resid, Prior=Priors)
# %% Visualization
plt.figure()
plt.hist(np.round(Pred_Cliped_label),
         bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.grid()
plt.title('Histogram of Rounded Predicted Labels')
plt.xlabel('Label')
plt.xlim((0.5, 5.5))

plt.figure()
plt.hist(Prior_Based_Prediction,
         bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.grid()
plt.title('Histogram of Prior-Based Predicted Labels')
plt.xlabel('Label')
plt.xlim((0.5, 5.5))

plt.figure()
plt.hist(Real_label,
         bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.grid()
plt.title('Histogram of Real Labels')
plt.xlabel('Label')
plt.xlim((0.5, 5.5))

# %% RMSE
print('****************************************')
print('Training notCliped RMSE')
print(np.sqrt(np.mean(np.square(Real_label - Pred_NotCliped_label))))
print('****************************************')
print('Training Cliped RMSE')
print(np.sqrt(np.mean(np.square(Real_label - Pred_Cliped_label))))
print('****************************************')
print('Training Rounded RMSE')
print(np.sqrt(np.mean(np.square(Real_label - np.round(Pred_Cliped_label)))))
print('****************************************')
print('Training Prior-Based RMSE')
print(np.sqrt(np.mean(np.square(Real_label - Prior_Based_Prediction))))

# %% Loading Test Data
file_path = "Data/sample_submission.csv"
data_test = utils.load_data_desired(file_path)

# %% Test Prediction
Pred_Test_NotCliped = []
Pred_Test_Cliped = []

for line in data_test:
    Pred_Test_NotCliped.append(alg.predict(str(line[1]),
                                           str(line[0]), clip=False).est)
    Pred_Test_Cliped.append(alg.predict(str(line[1]),
                                        str(line[0]), clip=True).est)

Pred_Test_NotCliped = np.array(Pred_Test_NotCliped)
Pred_Test_Cliped = np.array(Pred_Test_Cliped)

# %% Prior Based
Prior_Based_Pred_Test = np.zeros((len(Pred_Test_NotCliped), 5))

for i in range(5):
    Prior_Based_Pred_Test[:, i] = Priors[i] *\
        np.exp(-np.power((Pred_Test_NotCliped - (i + 1)),2)\
               / (2 * Sigma_Resid))

Prior_Based_Pred_Test = np.argmax(Prior_Based_Pred_Test, axis=1) + 1

# %% Visualization
plt.figure()
plt.hist(np.round(Pred_Test_Cliped),
         bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.grid()
plt.title('Histogram of Rounded Predicted Labels - Test')
plt.xlabel('Label')
plt.xlim((0.5, 5.5))

plt.figure()
plt.hist(Prior_Based_Pred_Test,
         bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.grid()
plt.title('Histogram of Prior-Based Predicted Labels - Test')
plt.xlabel('Label')
plt.xlim((0.5, 5.5))


# %% Save Prediction
file = open("testfile.csv", "w")
file.write("Id,Prediction\n")

for i in range(len(Pred_Test_NotCliped)):
    line = data_test[i]
    temp = 'r' + str(line[0]) + '_c' + str(line[1])\
        + ',' + str(int(Prior_Based_Pred_Test[i])) + '\n'
    file.write(temp)

file.close()


# %% Save Prediction
file = open("testfile2.csv", "w")
file.write("Id,Prediction\n")

for i in range(len(Pred_Test_Cliped)):
    line = data_test[i]
    temp = 'r' + str(line[0]) + '_c' + str(line[1]) +\
        ',' + str(int(round(Pred_Test_Cliped[i]))) + '\n'
    file.write(temp)

file.close()
