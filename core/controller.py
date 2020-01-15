import pandas as pd
import copy
import numpy as np
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from abc import ABCMeta
from .signal_manipulation import SmoothLibrary, AnomaliesLibrary, get_random_percent
from core.signal import Signal


class BaseSignalController(metaclass=ABCMeta):
    """Represents a signal, methods for signal processing, anomaly generation"""
    smooth_methods = ['savgol', 'moving_average', 'exponential', 'double_exponential', 'lowess']

    _normalize = False
    _scale = False
    _smooth = False

    control_results = None
    scaled_control_results = None
    smoothed_control_results = None

    target_variable = None

    def __init__(self, filepath, rolling_window_size=500, minimal_anomaly_length=50, sample_rate=40, encoding="cp1251",
                 delimiter=",", corr_threshold=0.9, smooth_method='savgol', target_variable=None):
        """ Class for preprocessing file with signal information, anomaly generation, smoothing signal,


        :param filepath: file to csv with signals information
        :param rolling_window_size: window length for signal cropping
        :param minimal_anomaly_length: minimal length of generated anomaly
        :param sample_rate: shift anomaly generation
        :param encoding: file reading encoding
        :param delimiter: csv file delimiter between columns signal dataset
        :param corr_threshold: minimum threshold of correlation value for feature removing
        :param smooth_method: 'savgol', 'moving_average', 'exponential', 'double_exponential', 'lowess'
        :param target_variable: target variable for usage without anomaly generation

        """
        super(BaseSignalController, self).__init__()
        self.filepath = filepath
        self.rolling_window_size = rolling_window_size
        self.sample_rate = sample_rate
        self.minimal_anomaly_length = minimal_anomaly_length
        # TODO different type of files (API) for reading
        self._read_signals_from_csv(encoding, delimiter)
        # TODO different pre-preprocessing statements
        self._preprocess_control_results(corr_threshold)
        # TODO different variations of signal scaling
        self._scale_signal()
        # TODO API access to signal scaling parameteres
        if smooth_method in self.smooth_methods:
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

    def _read_signals_from_csv(self, encoding, delimiter):
        """ Upload signal file from csv into memory
        :param encoding: (cp1251, utf-8, etc)
        :param delimiter: delimiter between columns in dataset
        """
        self.control_results = pd.read_csv(self.filepath, encoding=encoding, delimiter=delimiter)
        print("Number of columns:", len(self.control_results.columns))

    def _correlation_analysis(self, corr_threshold):
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
                StandardScaler().fit_transform(self.control_results))
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
    def get_sliced_signal(self):
        """ Property return sliced signals by params passed in the constructor
        :return: list Signal()
        """
        signal_samples = list()
        cut = True
        left_signal_border = 0
        right_signal_border = self.rolling_window_size
        while cut:
            if right_signal_border > len(self.smoothed_control_results.to_numpy()) or \
                    left_signal_border > len(self.smoothed_control_results.to_numpy()):
                cut = False
                continue
            signal_window = self.smoothed_control_results.to_numpy()[left_signal_border: right_signal_border, :]
            signal = Signal(signal_window)
            signal_samples.append(signal)
            left_signal_border += self.sample_rate
            right_signal_border += self.sample_rate
        return signal_samples

    def generate_anomalies(self, slice_signal):
        """

        :param slice_signal:
        :return:
        """
        signal_samples = copy.deepcopy(slice_signal)
        anomaly_signal_samples = list()
        funcs = [AnomaliesLibrary.change_trend]
        for signal in signal_samples:
            begin_index = int(np.random.randint(self.rolling_window_size - 2))
            end_index = int(np.random.randint(begin_index + 1, self.rolling_window_size))
            anomaly_length = end_index - begin_index

            while anomaly_length < self.minimal_anomaly_length:
                begin_index = int(np.random.randint(self.rolling_window_size - 2))
                end_index = int(np.random.randint(begin_index + 1, self.rolling_window_size))
                anomaly_length = end_index - begin_index

            anomaly_function = np.random.choice(funcs)
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


class SignalController(BaseSignalController):

    def __init__(self, *args, **kwargs):
        super(SignalController, self).__init__(*args, **kwargs)

    def _preprocess_control_results(self, *args):
        """

        :param args:
        :return:
        """
        super(SignalController, self)._preprocess_control_results(*args)
        self.control_results['Data'] = pd.to_datetime(self.control_results['Data'])
        self.control_results['Iter_Data'] = self.control_results['Data']
        self.control_results = self.control_results.set_index('Data')
        print("NaN Values:", self.control_results.isna().any().any())
        self.control_results = self.control_results.drop(['Iter_Data'], axis=1)


class GasolineSignalController(BaseSignalController):

    def __init__(self, *args, **kwargs):
        super(GasolineSignalController, self).__init__(*args, **kwargs)

    def _preprocess_control_results(self, *args):
        print('Read Gasoline')
        self.control_results = self.control_results.drop(['Time'], axis=1)
        super(GasolineSignalController, self).read_control_results(*args)
