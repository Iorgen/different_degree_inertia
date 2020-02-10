import pandas as pd
import copy
import os
import numpy as np
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler, Normalizer
from abc import ABCMeta
from seriem_temporis.exceptions.base_exceptions import ConfigException
from ..signal_manipulation import SmoothLibrary, AnomaliesLibrary, get_random_percent
from ..signal import Signal
from ..exceptions.base_exceptions import GeneratorException


class AbstractSignalHandler(metaclass=ABCMeta):
    """ Define high abstraction unit for data handler
        Here we declare configuration assign as private variables
    """

    def __init__(self, handler_conf):
        super(AbstractSignalHandler, self).__init__()
        try:
            self.__name = handler_conf['name']

            #
            self.__generate_anomaly_mode = handler_conf['generate_anomaly_mode']
            self.__anomaly_functions = handler_conf['anomaly_functions']
            self.__rolling_window_size = handler_conf['rolling_window_size']
            self.__minimal_anomaly_length = handler_conf['minimal_anomaly_length']
            self.__sample_rate = handler_conf['sample_rate']

            #
            self.__folder_path = handler_conf['folder_path']
            self.__delimiter = handler_conf['delimiter']
            self.__encoding = handler_conf['encoding']

            #
            self.__target_variable = handler_conf['target_variable']
            self.__target_values = handler_conf['target_values']

            #
            self.__smooth_method = handler_conf['smooth_method']
            self.__normalize = handler_conf['normalize']
            self.__scale = handler_conf['scale']
            self.__smooth = handler_conf['smooth']

        except KeyError as e:
            raise ConfigException(e, 'Configuration error')

    def get_batch_size(self):
        return 1 # return a tuple for model initialization


class TemplateSignalHandler(AbstractSignalHandler):

    def __init__(self, *args, **kwargs):
        super(TemplateSignalHandler, self).__init__(*args, **kwargs)

    def prepare_dataset(self):
        # here should be reading from dataset
        pass

    def read_signals_from_csv(self, filename, encoding, delimiter):
        """ Upload signal file from csv into memory
        :param filename: path to concrete file with signal
        :param encoding: (cp1251, utf-8, etc)
        :param delimiter: delimiter between columns in dataset
        """
        control_results = pd.read_csv(filename, encoding=encoding, delimiter=delimiter)
        print("Number of columns:", len(control_results.columns))
        return control_results

    def batch_generator(self):
        for filename in os.listdir(self.__folder_path):
            control_results = self.read_signals_from_csv(filename, self.__encoding, self.__delimiter)

            sliced_signals = set_part.get_sliced_signal
            shuffle(sliced_signals)
            input_train = np.array([signal.values for signal in sliced_signals])
            output_train = np.array([signal.condition_window for signal in sliced_signals])
            output_train = output_train.reshape((output_train.shape[0], output_train.shape[1], 1))
            output_train = to_categorical(output_train, num_classes=2)
            print('Input shape: {} \nOutput shape: {}'.format(input_train.shape, output_train.shape))
            yield input_train, output_train
