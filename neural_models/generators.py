import os
from seriem_temporis.controller import KasperskySetSignalController
from random import shuffle, randint
import numpy as np
from keras.utils import to_categorical


def dataset_reader(folder_path):
    for filename in os.listdir(folder_path):
        set_part = KasperskySetSignalController(
            filepath=folder_path + filename,
            rolling_window_size=500,
            minimal_anomaly_length=50,
            sample_rate=40,
            encoding="cp1251",
            delimiter=",",
            corr_threshold=0.9,
            smooth_method='savgol',
            generate_anomaly_mode=False,
            target_variable=55)
        sliced_signals = set_part.get_sliced_signal
        shuffle(sliced_signals)
        input_train = np.array([signal.values for signal in sliced_signals])
        output_train = np.array([signal.condition_window for signal in sliced_signals])
        output_train = output_train.reshape((output_train.shape[0], output_train.shape[1], 1))
        output_train = to_categorical(output_train, num_classes=2)
        print('Input shape: {} \nOutput shape: {}'.format(input_train.shape, output_train.shape))
        yield input_train, output_train
