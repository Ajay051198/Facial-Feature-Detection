""" This file would be used as a preprocessing step to load the data from the
.csv files and save it in a format which can be fed into out neural network"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def prepare_data(t_v):
    if t_v == "train":
        dir = 'data/training/training.csv'
    elif t_v == "validation":
        dir = 'data/test/test.csv'
    else:
        print('Invalid set selection')
        exit()

    # read the data into a pandas dataframe
    df = pd.read_csv(dir)

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # perform normalization
    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(X.shape[0],96,96,1)
    y = df.drop(['Image'], axis=1)
    y = y.to_numpy()

    return X, y
