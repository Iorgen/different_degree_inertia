from numpy.random import choice
import pandas as pd
import numpy as np
import math


def get_components(x, roll=30):
    s = pd.Series(x.reshape(-1))

    trend = s.rolling(roll).median().values
    noise = s - trend

    return trend, noise


class SmoothLibrary:

    @staticmethod
    def exponential_smoothing(series, alpha):
        result = [series[0]]
        for n in range(1, len(series)):
            result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
        return result

    @staticmethod
    def double_exponential_smoothing(series, alpha, beta):
        result = [series[0]]
        for n in range(1, len(series)):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series):  # прогнозируем
                value = result[-1]
            else:
                value = series[n]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)
        return result


class AnomaliesLibrary:

    @staticmethod
    def add_noise(data, begins, ends, percents=30, roll=30, source=None):
        """
        :param data: np.array
        :param begins: list of ints
        :param ends: list of ints
        :param percents: int or list of ints
        :param roll: int
        :return: 2 np.array
        """
        res = data.copy()
        mask = np.zeros_like(res)
        trend, noise = get_components(res, roll=roll)

        if source is None:
            source = data.copy()

        if isinstance(percents, int):
            percents = [percents] * len(begins)

        for b, e, p in zip(begins, ends, percents):
            noise_new = np.random.normal(0, source[b:e].std() * (1 + p / 100), size=e - b)
            for i in range(e - b):
                res[b + i] = trend[b + i] + noise_new[i]
                if math.isnan(res[b + i]):
                    res[b + i] = 1 + noise_new[i]
            mask[b: e] = 1
        return res, mask

    @staticmethod
    def change_trend(data, begins, ends, percents=10, roll=30, source=None):
        """
        :param data: np.array
        :param begins: list of ints
        :param ends: list of ints
        :param percents: int or list of ints
        :param roll: int
        :return: 2 np.array
        """
        res = data.copy()
        mask = np.zeros_like(res)

        if source is None:
            source = data.copy()

        if isinstance(percents, int):
            percents = [percents] * len(begins)

        for b, e, p in zip(begins, ends, percents):
            alpha = p / 1000 * source.std() * choice([-1, 1])
            for i in range(e - b):
                res[b + i] += i * alpha
            mask[b: e] = 1
        return res, mask

    @staticmethod
    def dome(data, begins, ends, percents=10, roll=30, source=None):
        """
        :param data: np.array
        :param begins: list of ints
        :param ends: list of ints
        :param percents: int or list of ints
        :param roll: int
        :return: 2 np.array
        """
        res = data.copy()
        mask = np.zeros_like(res)

        if source is None:
            source = data.copy()

        if isinstance(percents, int):
            percents = [percents] * len(begins)

        for b, e, p in zip(begins, ends, percents):
            alpha = p / 1000 * source.std() * choice([-1, 1])
            half = (e - b) // 2
            for i in range(half + 1):
                res[b + i] += i * alpha
                res[e - i] -= -i * alpha
            mask[b: e] = 1
        return res, mask

    @staticmethod
    def increase_dispersion(data, begins, ends, percents=100, roll=30, source=None):
        """
        :param data: np.array
        :param begins: list of ints
        :param ends: list of ints
        :param percents: int or list of ints
        :param roll: int
        :return: 2 np.array
        """
        res = data.copy()
        mask = np.zeros_like(res)
        trend, noise = get_components(res, roll=roll)

        if source is None:
            source = data.copy()

        if isinstance(percents, int):
            percents = [percents] * len(begins)

        for b, e, p in zip(begins, ends, percents):
            alpha = 1 + p / 100
            for i in range(e - b):
                res[b + i] = trend[b + i] + noise[b + i] * alpha
            mask[b: e] = 1
        return res, mask

    @staticmethod
    def decrease_dispersion(data, begins, ends, percents=100, roll=30, source=None):
        """
        :param data: np.array
        :param begins: list of ints
        :param ends: list of ints
        :param percents: int or list of ints
        :param roll: int
        :return: 2 np.array
        """
        res = data.copy()
        mask = np.zeros_like(res)
        trend, noise = get_components(res, roll=roll)

        if source is None:
            source = data.copy()

        if isinstance(percents, int):
            percents = [percents] * len(begins)

        for b, e, p in zip(begins, ends, percents):
            alpha = max(1 - p / 100, 0)
            for i in range(e - b):
                res[b + i] = trend[b + i] + noise[b + i] * alpha
            mask[b: e] = 1
        return res, mask

    @staticmethod
    def shift_trend(data, begins, ends, percents=5, roll=30, source=None):
        """
        :param data: np.array
        :param begins: list of ints
        :param ends: list of ints
        :param percents: int or list of ints
        :param roll: int
        :return: 2 np.array
        """
        res = data.copy()
        mask = np.zeros_like(res)

        if source is None:
            source = data.copy()

        if isinstance(percents, int):
            percents = [percents] * len(begins)

        for b, e, p in zip(begins, ends, percents):
            shift = source.mean() * p / 100 * choice([-1, 1])
            for i in range(e - b):
                res[b + i] += shift
            mask[b: e] = 1
        return res, mask
