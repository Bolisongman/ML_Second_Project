# *****************************************************************************
# %% Load Packages
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import model_selection

import random
import numpy as np

import matplotlib.pyplot as plt

import utils

# *****************************************************************************
# %% Random Seed
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

# *****************************************************************************
# %% Load Project Dataset in a Surprise format
file_path = "Data/data_train_Surprise_format.csv"

# Defining the reader file for the surprise package
reader = Reader(line_format='item user rating', sep=',',
                rating_scale=(1, 5), skip_lines=1)

data_train = Dataset.load_from_file(file_path, reader=reader)

# *****************************************************************************
# %% Hyper parameter tuning and CV analysis
# Algorithm: SVD

# Algorithm: Set of Hyper-parameters in which we want to search
Hyper_Params = {'n_epochs': [300],
                'n_factors': [25],
                'biased': [True],
                'lr_all': [0.005],
                'reg_pu': [0.1],
                'reg_qi': [0.1],
                'reg_bi': [0.3],
                'reg_bu': [0.01],
                'verbose': [True]}

# Defining an object for searching over the hyper-parameter space and applying
# 3-fold cross-validation for each elements
Train_CV = model_selection.GridSearchCV(SVD,
                                        Hyper_Params,
                                        measures=['rmse'],
                                        cv=3, n_jobs=3)

# Training the defined object
Train_CV.fit(data_train)

# *****************************************************************************
# %% Figures
# Ploting RMSE vs lambda bi
plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_reg_bi'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$)')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$)')
plt.savefig('3_fold_CV_Reg_Param_bi.png')

# Ploting RMSE vs lambda bu
plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_reg_bu'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$)')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$)')
plt.savefig('3_fold_CV_Reg_Param_bu.png')

# Ploting RMSE vs lambda pu
plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_reg_pu'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$)')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$)')
plt.savefig('3_fold_CV_Reg_Param_pu.png')

# Ploting RMSE vs lambda qi
plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_reg_qi'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xscale('log')
plt.xlabel('Regularization Parameter ($\lambda$)')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Regularization Parameter ($\lambda$)')
plt.savefig('3_fold_CV_Reg_Param_qi.png')

# Ploting RMSE vs K (number of latent variables\factors)
plt.figure(figsize=(20, 12))
plt.rcParams.update({'font.size': 12})
plt.plot(Train_CV.cv_results['param_n_factors'],
         Train_CV.cv_results['mean_test_rmse'], '.k')
plt.xlabel('Number of Factores')
plt.ylabel('RMSE')
plt.grid()
plt.title('3-Fold CV - Number of Factors')
plt.savefig('3_fold_CV_Factors.png')

# *****************************************************************************
# %% Save the detailed results of Grid Search with name Details.txt
file = open("SVD_Details.txt", "w")

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

# *****************************************************************************
# %% Best Hyper-parameters Training:
# Training over whole training dataset, using best hyper-parameters
alg = SVD()

alg.biased = Train_CV.best_params['rmse']['biased']
alg.n_epochs = Train_CV.best_params['rmse']['n_epochs']
alg.n_factors = Train_CV.best_params['rmse']['n_factors']
alg.reg_pu = Train_CV.best_params['rmse']['reg_pu']
alg.reg_qi = Train_CV.best_params['rmse']['reg_qi']
alg.reg_bu = Train_CV.best_params['rmse']['reg_bu']
alg.reg_bi = Train_CV.best_params['rmse']['reg_bi']
alg.lr_pu = Train_CV.best_params['rmse']['lr_all']
alg.lr_qi = Train_CV.best_params['rmse']['lr_all']
alg.verbose = True
alg.random_state = 0

alg.fit(data_train.build_full_trainset())

# *****************************************************************************
# %% Loading Test Data
file_path = "Data/sample_submission.csv"
data_test = utils.load_data_desired(file_path)

# *****************************************************************************
# %% Predicting test data labels
Predict_Test = []

for line in data_test:
    Predict_Test.append(alg.predict(str(line[1]), str(line[0])).est)

# *****************************************************************************
# %% Save Predictions as a submission file
file = open("SVD_Submission_file.csv", "w")
file.write("Id,Prediction\n")

for i in range(len(Predict_Test)):
    line = data_test[i]
    temp = 'r' + str(line[0]) + '_c' + str(line[1]) + ',' + str(int(round(Predict_Test[i]))) + '\n'
    file.write(temp)

file.close()
