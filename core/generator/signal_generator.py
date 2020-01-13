from core.signal import Signal
import numpy as np
from ..signal_manipulation import AnomaliesLibrary


def get_random_percent(name):
    if name == 'change_trend':
        return np.random.uniform(6, 12)
    elif name == 'dome':
        return np.random.uniform(8, 14)
    elif name == 'increase_dispersion':
        return np.random.uniform(150, 190)
    elif name == 'decrease_dispersion':
        return np.random.uniform(70, 100)
    elif name == 'shift_trend':
        return np.random.uniform(10, 18)
    elif name == 'add_noise':
        return np.random.uniform(1, 7)


class OneSampleAnomalyGenerator:

    def __init__(self, signals, rolling_window_size=500, sample_rate=40, minimal_anomaly_length=50):
        assert isinstance(signals, np.ndarray)
        self.signals = signals
        self.sample_rate = sample_rate
        self.rolling_window_size = rolling_window_size

        # Параметр подвергаемый масштабированию
        self.anomaly_numbers = 1
        self.signal_samples = list()
        self.anomaly_signal_samples = list()
        self.minimal_anomaly_length = minimal_anomaly_length

    def slice_signals(self):
        cut = True
        left_signal_border = 0
        right_signal_border = self.rolling_window_size
        while cut:
            if right_signal_border > len(self.signals) or left_signal_border > len(self.signals):
                cut = False
                continue
            signal_window = self.signals[left_signal_border: right_signal_border, :]
            print(signal_window.shape)
            signal = Signal(signal_window)
            self.signal_samples.append(signal)
            left_signal_border += self.sample_rate
            right_signal_border += self.sample_rate

    def generate_anomaly(self):
        funcs = [AnomaliesLibrary.add_noise, AnomaliesLibrary.shift_trend, AnomaliesLibrary.increase_dispersion]
        print("another generating noise anomaly type")
        for signal in self.signal_samples:
            percent = np.random.uniform(500, 700)
            begin_index = int(np.random.randint(self.rolling_window_size - 2))
            end_index = int(np.random.randint(begin_index + 1, self.rolling_window_size))
            anomaly_length = end_index - begin_index
            while anomaly_length < self.minimal_anomaly_length:
                begin_index = int(np.random.randint(self.rolling_window_size - 2))
                end_index = int(np.random.randint(begin_index + 1, self.rolling_window_size))
                anomaly_length = end_index - begin_index

            anomaly_function = np.random.choice(funcs)
            percent = get_random_percent(anomaly_function.__name__)
            initial_signal, abnormal_signal_part = anomaly_function(signal.values, begins=[begin_index],
                                                                    ends=[end_index],
                                                                    percents=[percent],
                                                                    source=signal.values)
            anomaly_signal = Signal(initial_signal, abnormal_signal_part, abnormal=True)
            self.anomaly_signal_samples.append(anomaly_signal)


class MultipleSamplesAnomalyGenerator:
    """ Base class saving signal with created anomalies parts
    """

    def __init__(self, signals, rolling_window_size=500, sample_rate=40, minimal_anomaly_length=50):
        assert isinstance(signals, np.ndarray)
        self.signals = signals
        self.sample_rate = sample_rate
        self.rolling_window_size = rolling_window_size

        # Параметр подвергаемый масштабированию
        self.anomaly_numbers = 1
        self.signal_samples = list()
        self.anomaly_signal_samples = list()
        self.minimal_anomaly_length = minimal_anomaly_length

    def slice_signals(self):
        pass

    def generate_anomaly(self):
        pass

