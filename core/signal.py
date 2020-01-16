import numpy as np


class Signal:
    """

    """
    def __init__(self, values, condition_window=None, abnormal=False):
        assert isinstance(values, np.ndarray)
        self.__predict_condition_window = None
        self.__predict_abnormal = None
        self.values = values
        self.abnormal = abnormal
        if condition_window is not None:
            self.condition_window = condition_window
        else:
            self.condition_window = [0] * len(self.values)

    @property
    def predict_condition_window(self):
        return self.__predict_condition_window

    @predict_condition_window.setter
    def predict_condition_window(self, value):
        self.__predict_condition_window = value

    @property
    def predict_abnormal(self):
        return self.__predict_abnormal

    @predict_abnormal.setter
    def predict_abnormal(self, value):
        self.__predict_abnormal = value
