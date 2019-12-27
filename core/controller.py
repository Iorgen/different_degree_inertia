import pandas as pd
import copy
import numpy as np
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from abc import ABCMeta
from .signal_manipulation import SmoothLibrary
from .signal import Signal


class BaseSignalController(metaclass=ABCMeta):
    _normalize = False
    _scale = False
    _smooth = False

    # Base non smoothed values from source
    control_results = None
    scaled_control_results = None

    # Smoothed values from source using savgol filter method
    savgol_filter_smoothing = None
    scaled_savgol_filter_smoothing = None

    # another smoothing values c
    moving_average_rolling = None
    exponential_smoothing_results = None
    double_exponential_smoothing_results = None
    lowess_smoothing_results = None

    def __init__(self, filepath, encoding, delimiter, corr_threshold=0.9):
        """

        :param filepath:
        :param encoding:
        :param delimiter:
        :param corr_threshold:
        """
        super(BaseSignalController, self).__init__()
        self.filepath = filepath
        self.read_control_results(encoding, delimiter, corr_threshold)
        self.smooth_savgol_filter()
        self.scale_signal()

    def read_control_results(self, encoding, delimiter, corr_threshold):
        """ Base function for reading control results from pandas, can be overloading in child classes
        :param encoding: (cp1251, utf-8, etc)
        :param delimiter: delimeter between columns in dataset
        :param corr_threshold: minimum threshold of correlation value for removing a feature
        """
        # Read using pandas library
        self.control_results = pd.read_csv(self.filepath, encoding=encoding, delimiter=delimiter)
        print("Number of columns:", len(self.control_results.columns))

        # drop nan values from dataset
        self.control_results = self.control_results.dropna(axis='columns')
        print("Number of columns after clean:", len(self.control_results.columns))

        # correlation check and drop correlated features
        corr_matrix = self.control_results.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        self.control_results = self.control_results.drop(to_drop, axis=1)
        print("Number of columns after corr analysis:", len(self.control_results.columns))

    def smooth_savgol_filter(self):
        """

        :return:
        """
        self.savgol_filter_smoothing = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.savgol_filter_smoothing[cont_res] = savgol_filter(self.control_results[cont_res], 15, 3)
        print("savgol filter smoothing successful")

    def smooth_moving_average(self, n):
        """

        :param n:
        :return:
        """
        self.moving_average_rolling = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.moving_average_rolling[cont_res] = self.control_results[cont_res].rolling(window=n).mean().fillna(0)

    def smooth_exponential(self, alpha):
        """

        :param alpha:
        :return:
        """
        self.exponential_smoothing_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.exponential_smoothing_results[cont_res] = SmoothLibrary.exponential_smoothing(
                self.control_results[cont_res], alpha)

    def smooth_double_exponential(self, alpha, beta):
        """

        :param alpha:
        :param beta:
        :return:
        """
        self.double_exponential_smoothing_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.double_exponential_smoothing_results[cont_res] = SmoothLibrary.double_exponential_smoothing(
                self.control_results[cont_res], alpha, beta)

    def smooth_lowess(self):
        """

        :return:
        """
        self.lowess_smoothing_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.lowess_smoothing_results[cont_res] = lowess(self.control_results[cont_res],
                                                             range(0, len(self.control_results[cont_res])),
                                                             it=0, frac=0.02, is_sorted=True)

    def scale_signal(self):
        """

        :return:
        """
        if self.control_results is not None:
            self.scaled_control_results = pd.DataFrame(
                StandardScaler().fit_transform(self.control_results))
            print('successfully scaled control_results')
        else:
            print('There is no control_results')
        if self.savgol_filter_smoothing is not None:
            self.scaled_savgol_filter_smoothing = pd.DataFrame(
                StandardScaler().fit_transform(self.savgol_filter_smoothing))
            print('successfully scaled savgol filter control results')
        else:
            print('There is no scaled_control_results')


class SignalController(BaseSignalController):

    def __init__(self, *args, **kwargs):
        super(SignalController, self).__init__(*args, **kwargs)

    def read_control_results(self, *args):
        super(SignalController, self).read_control_results(*args)
        self.control_results['Data'] = pd.to_datetime(self.control_results['Data'])
        self.control_results['Iter_Data'] = self.control_results['Data']
        self.control_results = self.control_results.set_index('Data')
        print("NaN Values:", self.control_results.isna().any().any())
        self.control_results = self.control_results.drop(['Iter_Data'], axis=1)


class GasolineSignalController(BaseSignalController):

    def __init__(self, *args, **kwargs):
        super(GasolineSignalController, self).__init__(*args, **kwargs)

    def read_control_results(self, *args):
        print('Read Gasoline ')
        super(GasolineSignalController, self).read_control_results(*args)
