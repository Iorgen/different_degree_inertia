import numpy as np


class Signal:

    def __init__(self, values, condition_window=None, abnormal=False):
        assert isinstance(values, np.ndarray)
        self.values = values
        self.abnormal = abnormal
        # TODO rename into condition_window
        if condition_window is not None:
            self.condition_window = condition_window
        else:
            self.condition_window = [0] * len(self.values)
