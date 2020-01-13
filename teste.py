import numpy as np
import pandas as pd
import copy
import os

import matplotlib.pyplot as plt
import seaborn as sns

from random import shuffle, randint
from numpy import mean, std, dstack

from pandas import read_csv

from keras import backend as K
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Flatten, Dropout, Conv1D, LSTM, GRU,
                          TimeDistributed, GlobalAveragePooling1D, MaxPooling1D)
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

from keras.optimizers import Adam, RMSprop,SGD
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from scipy import signal
from core.controller import SignalController
from core.generator.signal_generator import OneSampleAnomalyGenerator


if __name__ == "__main__":
    signals = SignalController(filepath="datasets/DATA/Nikol/L1_10-12.18.csv")
    signal_samples = signals.get_sliced_signal
    signals.generate_anomalies()

    print(signal_samples)
