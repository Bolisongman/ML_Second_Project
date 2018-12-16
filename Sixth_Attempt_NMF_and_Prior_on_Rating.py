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
file_path = "Data/data_train_preprocessed.csv"
data_train = utils.load_data_desired(file_path)

# %% 















