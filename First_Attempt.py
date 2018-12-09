#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 15:11:59 2018

@author: alireza
"""

# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import model_selection

import random
import numpy as np

# %% Random Seed
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

# %% Load Project Dataset in a Surprise format
# Path to dataset file
file_path = "Data/data_train_Surprise_format.csv"

# Reader format
reader = Reader(line_format='item user rating', sep=',',
                rating_scale=(1, 5), skip_lines=1)

# Loading training data
data_train = Dataset.load_from_file(file_path, reader=reader)

# %% Hyper parameter tuning and CV analysis
# Algorithm: SVD
Hyper_Params = {'n_epochs': [10],
                'n_factors': [50, 100, 150, 200],
                'biased': [False],
                'lr_all': [0.005],
                'reg_all': [0.01, 0.1, 0.3, 1.0]}

Train_CV = Grid_Search_Result = model_selection.GridSearchCV(SVD,
                                                             Hyper_Params,
                                                             measures=['rmse', 'mae'],
                                                             cv=3, n_jobs=3)

Train_CV.fit(data_train)
