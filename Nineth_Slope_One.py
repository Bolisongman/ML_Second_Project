#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:29:02 2018

@author: alireza
"""
# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import SlopeOne
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

# %% 3-fold CV
start = time.time()

algo = SlopeOne()
model_selection.cross_validate(algo, data_train, measures=['RMSE'], cv=3,
                               verbose=True, n_jobs=3)

end = time.time()
print("***********************************************")
print("Exe time:")
print(end - start)



