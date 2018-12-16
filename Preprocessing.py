#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:15:05 2018

@author: alireza
"""


# %% Load Packages
import random
import numpy as np
import utils

# %% Random Seed
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

# %% Delete Outliers
file_path = "Data/data_train.csv"
save_path01 = "Data/data_train_preprocessed_500.csv"

labels = utils.delete_users(file_path, min_num_items=500,
                      save_path=save_path01)

# %% Save in the Surprise format
save_path02 = "Data/data_train_preprocessed_surprise_format_500.csv"
utils.convert_to_surprise_format(
        load_path=save_path01,
        save_path=save_path02)
