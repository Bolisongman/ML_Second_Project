# *****************************************************************************
import scipy.sparse as sp
import numpy as np
import csv


# *****************************************************************************
# %% The function written in the helpers.py for Excersise 10
def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset):
    """Load data in text format, one rating per line."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


# *****************************************************************************
# %% Load data in a desirable form for Surprise
def deal_line(line):
    """extracting the information from a line of datasets in the format of
       the project datasets."""
    pos, rating = line.split(",")
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)


def load_data_desired(path_dataset):
    """Load data in text format, one rating per line."""
    data = read_txt(path_dataset)[1:]
    data = [deal_line(line) for line in data]
    return data


# *****************************************************************************
# %% A function for converting a project-csv file to surprise dataset format
def convert_to_surprise_format(load_path, save_path='surprise_data.csv'):
    """Converting a dataset with project format to a dataset with the structure
        of Surprise package.
    Args:
        * load path
        * save path
    Returns:
        * -
    """
    data = load_data_desired(load_path)
    file = open(save_path, "w")
    file.write("Movie,Subject,Pediction\n")

    for line in data:
        temp = str(line[0]) + ',' + str(line[1]) + ',' + str(int(line[2])) + '\n'
        file.write(temp)

    file.close()


# *****************************************************************************
# %% A function for Prior-correcting of prediction
def Prior_Correction(Raw_Prediction, Noise_Var=1, Prior=np.ones(5)/5):
    """Estimating the integer rating using the predicted continous one
    Args:
        * Raw_Prediction: continous prediction of ratings
        * Nois_Var: variance of noise
        * Prior: prior distribution over integer ratings
    Returns:
        * Integer predictions
    """
    Prior_Based_Pred = np.zeros((len(Raw_Prediction), 5))

    for i in range(5):
        Prior_Based_Pred[:, i] = Prior[i] *\
            np.exp(-np.power((Raw_Prediction - (i + 1)), 2) / (2 * Noise_Var))

    Prior_Based_Pred = np.argmax(Prior_Based_Pred, axis=1) + 1
    return Prior_Based_Pred

# *****************************************************************************
# %% Building K-ind for K-fold CV
# This function had been already written by TAs for one of the Lab exersices
def build_k_indices(y, k_fold, seed):
    """Generating k set of indexes for K-fold CV.
    Args:
        * y: labels
        * k_fold: number of folds
        * seed: random seed
    Returns:
        * List of k_fold arrays of indexes
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


# *****************************************************************************
# %%  preprocessing data, makes a csv file with the new data and returns the
# indices of this new data.
def delete_users(path_dataset, min_num_items, num_users=1000,
                 save_path='preprocessed_data.csv'):
    """eliminates the inactive users from the original dataset.
    Args:
        * min_num_items:
            all users we keep should have rated at least min_num_items item.
        * save_path:
            path of output file
    Returns:
        * a csv file of the preprocesses data in the same format with
            the original data - saved as preprocessed_data.csv
        * labels:
            keeps the indices of the entries in the new csv file with respect
            to the original data.
    """

    # Loading Data
    data = read_txt(path_dataset)[1:]
    data2 = [deal_line(line) for line in data]

    # Counting number of movies wathced by a person
    cnt_items_per_user = np.zeros(num_users)
    for row, col, rating in data2:
        cnt_items_per_user[col-1] += 1

    # Generating cleaned data set: label==-1 <-> the user is removed
    data_new = []
    labels = (-1) * np.ones(len(data))

    cnt_deleted = 0
    for i in range(len(data)):
        if(cnt_items_per_user[data2[i][1]-1] < min_num_items):
            cnt_deleted += 1
        else:
            data_new.append(data[i])
            labels[i] = int(i - cnt_deleted)

    # Save Output
    file = open(save_path, "w")
    file.write("Id,Pediction\n")

    for line in data_new:
        file.write(line + "\n")

    file.close()
    return labels
