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

from seriem_temporis.controller import SignalController, KasperskySetSignalController


if __name__ =="__main__":
    signal_samples = list()
    for filename in os.listdir('/media/jurgen/Новый том/Sygnaldatasets/kaspersky/attacks/')[1:3]:
        signals = KasperskySetSignalController(
            filepath='/media/jurgen/Новый том/Sygnaldatasets/kaspersky/attacks/' + filename,
            rolling_window_size=500,
            minimal_anomaly_length=50,
            sample_rate=40,
            encoding="cp1251",
            delimiter=",",
            corr_threshold=0.9,
            smooth_method='savgol',
            generate_anomaly_mode=False,
            target_variable=55)

        signal_samples.extend(signals.get_sliced_signal)

    print(len(signal_samples))
