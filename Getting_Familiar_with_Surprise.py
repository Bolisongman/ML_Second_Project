#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 13:22:39 2018

@author: alireza
"""
# %% Load Packages
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

import numpy as np
import matplotlib.pyplot as plt

# %% Test the package code
# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# %% Load Project Dataset

