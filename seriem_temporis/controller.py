import copy
from abc import ABCMeta

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import Normalizer
from statsmodels.nonparametric.smoothers_lowess import lowess

from .exceptions.base_exceptions import GeneratorException
from .signal import Signal
from .signal_manipulation import SmoothLibrary, get_random_percent


class BaseSignalHandler(metaclass=ABCMeta):
    """Represents a signal, methods for signal processing, anomaly generation """
    __filepath = None
    __rolling_window_size = None
    __minimal_anomaly_length = None
    __sample_rate = None
    __generate_anomaly_mode= None
    __target_variable = None

    smooth_methods = ['savgol', 'moving_average', 'exponential', 'double_exponential', 'lowess']

    _normalize = False
    _scale = False
    _smooth = False

    control_results = None
    scaled_control_results = None
    smoothed_control_results = None

    _target_variable = None
    _target_values = None

    def __init__(self,
                 filepath,
                 rolling_window_size=500,
                 minimal_anomaly_length=50,
                 sample_rate=40,
                 corr_threshold=0.9,
                 smooth_method='savgol',
                 generate_anomaly_mode=True,
                 target_variable=None,
                 delimiter=",", encoding="cp1251"
                 ):
        super(BaseSignalHandler, self).__init__()
        self.__filepath = filepath
        self.__rolling_window_size = rolling_window_size
        self.__minimal_anomaly_length = minimal_anomaly_length
        self.__sample_rate = sample_rate
        self.__generate_anomaly_mode = generate_anomaly_mode
        self.__target_variable = target_variable
        # TODO different type of files (API) for reading
        self._read_signals_from_csv(encoding, delimiter)
        # TODO emission filling
        if not self.__generate_anomaly_mode and self.__target_variable:
            self._target_values = self.control_results[[self.__target_variable]].to_numpy().astype(int)
        # TODO different pre-preprocessing statements
        self._preprocess_control_results(corr_threshold)
        # TODO different variations of signal scaling
        self._scale_signal()
        # TODO API access to signal scaling parameteres
        # TODO return error when smooth method is wrong
        if smooth_method in self.smooth_methods:
            self.__smooth_method = smooth_method
            if smooth_method == 'savgol':
                self.smooth_using_savgol_filter()
            elif smooth_method == 'moving_average':
                self.smooth_using_moving_average(10)
            elif smooth_method == 'exponential':
                self.smooth_using_exponential_method(10)
            elif smooth_method == 'double_exponential':
                self.smooth_using_double_exponential_method(10)
            elif smooth_method == 'lowess':
                self.smooth_using_lowess()


    def load_configuration_from_class_object(self, configuration):
        pass

    def load_configuration_from_json(self):
        pass

    def smooth_signal(self):
        """

        :return:
        """
        if self.__smooth_method == 'savgol':
            self.smooth_using_savgol_filter()
        elif self.__smooth_method == 'moving_average':
            self.smooth_using_moving_average(10)
        elif self.__smooth_method == 'exponential':
            self.smooth_using_exponential_method(10)
        elif self.__smooth_method == 'double_exponential':
            self.smooth_using_double_exponential_method(10)
        elif self.__smooth_method == 'lowess':
            self.smooth_using_lowess()

    def _read_signals_from_csv(self, encoding, delimiter):
        """ Upload signal file from csv into memory
        :param encoding: (cp1251, utf-8, etc)
        :param delimiter: delimiter between columns in dataset
        """
        self.control_results = pd.read_csv(self.__filepath, encoding=encoding, delimiter=delimiter)
        print("Number of columns:", len(self.control_results.columns))

    def _correlation_analysis(self, corr_threshold):
        """ Method which compute correlation between features and then drop features with corr coef higher then
        corr_threshold
        :param corr_threshold: float -
        """
        corr_matrix = self.control_results.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        self.control_results = self.control_results.drop(to_drop, axis=1)
        print("Number of columns after corr analysis:", len(self.control_results.columns))

    def _preprocess_control_results(self, corr_threshold):
        """ Base function for reading control results from file,
        :param corr_threshold: minimum threshold of correlation value for feature removing
        """
        self._correlation_analysis(corr_threshold=corr_threshold)
        self.control_results = self.control_results.dropna(axis='columns')
        print("Number of columns after clean:", len(self.control_results.columns))

    def _scale_signal(self):
        """ Standardize features by removing the mean and scaling to unit variance
        The standard score of a sample x is calculated as:
        z = (x - u) / s
        """
        if self.control_results is not None:
            self.scaled_control_results = pd.DataFrame(
                Normalizer().fit_transform(self.control_results))
            print('Successfully scaled control_results')
        else:
            print('There is no control_results')
            # TODO raise exception if something go wrong

    def smooth_using_savgol_filter(self):
        """
        """
        self.smoothed_control_results = copy.deepcopy(self.scaled_control_results)
        for cont_res in self.scaled_control_results:
            self.smoothed_control_results[cont_res] = savgol_filter(self.scaled_control_results[cont_res], 15, 3)
        print("savgol filter smoothing successful")

    def smooth_using_moving_average(self, n):
        """
        :param n:
        """
        self.smoothed_control_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.smoothed_control_results[cont_res] = self.control_results[cont_res].rolling(window=n).mean().fillna(0)

    def smooth_using_exponential_method(self, alpha):
        """
        :param alpha:
        """
        self.smoothed_control_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.smoothed_control_results[cont_res] = SmoothLibrary.exponential_smoothing(
                self.control_results[cont_res], alpha)

    def smooth_using_double_exponential_method(self, alpha, beta):
        """
        :param alpha:
        :param beta:
        """
        self.smoothed_control_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.smoothed_control_results[cont_res] = SmoothLibrary.double_exponential_smoothing(
                self.control_results[cont_res], alpha, beta)

    def smooth_using_lowess(self):
        """
        """
        self.smoothed_control_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.smoothed_control_results[cont_res] = lowess(self.control_results[cont_res],
                                                             range(0, len(self.control_results[cont_res])),
                                                             it=0, frac=0.02, is_sorted=True)

    @property
    def get_class_balance(self):
        """ Property which return (anomaly capacity)/normal_samples
        1 mean balanced
        > 1 means more abnormal
        < 1 means less abnormal
        :return: int
        """

        return 1

    @property
    def get_sliced_signal(self):
        """ Property return sliced signals by params passed in the constructor
        :return: list Signal()
        """
        signal_samples = list()
        cut = True
        left_signal_border = 0
        right_signal_border = self.__rolling_window_size
        while cut:
            if right_signal_border > len(self.smoothed_control_results.to_numpy()) or \
                    left_signal_border > len(self.smoothed_control_results.to_numpy()):
                cut = False
                continue
            signal_window = self.smoothed_control_results.to_numpy()[left_signal_border: right_signal_border, :]
            if not self.__generate_anomaly_mode and self.__target_variable:
                condition_window = self._target_values[left_signal_border: right_signal_border]
                if 1 in condition_window:
                    abnormal = True
                else:
                    abnormal = False
                signal = Signal(signal_window, condition_window=condition_window, abnormal=abnormal)
            elif self.__generate_anomaly_mode:
                signal = Signal(signal_window)
            else:
                raise Exception
            signal_samples.append(signal)
            left_signal_border += self.__sample_rate
            right_signal_border += self.__sample_rate
        return signal_samples

    def generate_anomalies(self, slice_signal):
        """

        :param slice_signal:
        :return:
        """


        if not self.__generate_anomaly_mode:
            raise GeneratorException(self.__generate_anomaly_mode, 'Wrong generator mode')
        signal_samples = copy.deepcopy(slice_signal)
        anomaly_signal_samples = list()
        for signal in signal_samples:
            begin_index = int(np.random.randint(self.__rolling_window_size - 2))
            end_index = int(np.random.randint(begin_index + 1, self.__rolling_window_size))
            anomaly_length = end_index - begin_index

            while anomaly_length < self.__minimal_anomaly_length:
                begin_index = int(np.random.randint(self.__rolling_window_size - 2))
                end_index = int(np.random.randint(begin_index + 1, self.__rolling_window_size))
                anomaly_length = end_index - begin_index

            anomaly_function = np.random.choice(self.config.config_by_name('anomaly_functions'))
            percent = get_random_percent(anomaly_function.__name__)
            abnormal_signal_part = None

            for idx in range(signal.values.shape[1]):
                initial_signal, abnormal_signal_part = anomaly_function(signal.values[:, idx], begins=[begin_index],
                                                                        ends=[end_index],
                                                                        percents=[percent],
                                                                        source=signal.values[:, idx])
                signal.values[:, idx] = initial_signal
            anomaly_signal = Signal(signal.values, abnormal_signal_part, abnormal=True)
            anomaly_signal_samples.append(anomaly_signal)
        return anomaly_signal_samples


class SignalController(BaseSignalHandler):

    def __init__(self, *args, **kwargs):
        super(SignalController, self).__init__(*args, **kwargs)

    def _preprocess_control_results(self, *args):
        super(SignalController, self)._preprocess_control_results(*args)
        self.control_results['Data'] = pd.to_datetime(self.control_results['Data'])
        self.control_results['Iter_Data'] = self.control_results['Data']
        self.control_results = self.control_results.set_index('Data')
        print("NaN Values:", self.control_results.isna().any().any())
        self.control_results = self.control_results.drop(['Iter_Data'], axis=1)


class KasperskySetSignalController(BaseSignalHandler):

    def __init__(self, *args, **kwargs):
        super(KasperskySetSignalController, self).__init__(*args, **kwargs)

    def _read_signals_from_csv(self, encoding, delimiter):
        self.control_results = pd.read_csv(self.__filepath, encoding=encoding, delimiter=delimiter, header=None)
        print("Number of columns:", len(self.control_results.columns))

    def _correlation_analysis(self, corr_threshold):
        pass

    def _preprocess_control_results(self, *args):
        self.control_results = self.control_results.drop([0, 54, 55, 56, 57], axis=1)
        print("NaN Values:", self.control_results.isna().any().any())
        super(KasperskySetSignalController, self)._preprocess_control_results(*args)
