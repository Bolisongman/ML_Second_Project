#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:22:39 2018

@author: alireza
"""
# %% Load Packages
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

import numpy as np
import matplotlib.pyplot as plt

import utils

# %% Test the package code
# Load the movielens-100k dataset (download it if needed),
data_ml100k = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data_ml100k, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# %% Load Project Dataset
Path = "Data/data_train.csv"
Data_train = utils.load_data(Path)

# %% Load Project Dataset in a Surprise format
# path to dataset file
file_path = "Data/data_train_Surprise_format.csv"

reader = Reader(line_format='item user rating', sep=',',)
data = Dataset.load_from_file(file_path, reader=reader)
