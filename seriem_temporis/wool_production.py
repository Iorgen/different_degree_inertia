import sklearn
import numpy as np
import copy
import keras
import xlrd
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0)
plt.rcParams.update({'figure.max_open_warning': 0})
from numpy import mean
from numpy import std
from numpy import dstack
from scipy import signal, fftpack
from scipy.signal import savgol_filter
from pandas import read_csv
from matplotlib import pyplot
from random import shuffle, randint
import numpy
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
numpy.random.seed(7)
from datetime import datetime


# TODO Adapt this class under inheritance from BaseSignalController from controller module
class MineralWoolProductionSignal:
    values = None
    smoothed_values = None
    special_marking_data = None
    target = None
    dates = pd.DataFrame()

    def __init__(self, filepath="SOP1.txt"):
        self.filepath = filepath
        self.read_control_results()

    def read_control_results(self):
        self.values = pd.read_csv(self.filepath, encoding="cp1251", delimiter=",")
        print("Number of columns:", len(self.values.columns))
        self.values = self.values.dropna(axis='columns')
        print("Number of columns after clean:", len(self.values.columns))
        corr_matrix = self.values.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        self.values = self.values.drop(to_drop, axis=1)
        print("Number of columns after corr analysis:", len(self.values.columns))
        self.values['Data'] = pd.to_datetime(self.values['Data'])
        self.values['Iter_Data'] = self.values['Data']
        self.values = self.values.set_index('Data')
        print("NaN Values:", self.values.isna().any().any())
        self.values.info()

    def read_target(self):
        self.target = pd.read_csv('datasets/DATA/Nikol/plains_1.csv', delimiter=',')
        self.target = self.target.dropna()
        self.target = self.target.drop_duplicates(subset="END_DATE")
        self.target["test_index"] = 50
        self.target['START_DATE'] = pd.to_datetime(self.target['START_DATE'])
        self.target['END_DATE'] = pd.to_datetime(self.target['END_DATE'])
        self.target['Data'] = self.target['START_DATE']
        self.target = self.target.set_index('Data')
        self.target = self.target.sort_index()
        print("NaN Values:", self.target.isna().any().any())
        self.target.info()

    def clear_target(self):
        # target.loc[(target['REASON']!= 'Слив металла') & (target['REASON']!= 'Переход'), 'REASON'].size
        self.target = self.target.loc[(self.target['REASON'] != 'Слив металла') & (self.target['REASON'] != 'Переход'),
                                      ['EQUIPMENT', 'REASON', 'START_DATE', 'END_DATE', 'ALL_PLAINS_TIME',
                                       'test_index']]
        self.target['REASON'].value_counts()

        def parse_minutes(s):
            hours, minutes = s.split(':')
            return int(hours) * 60 + int(minutes)

        # Convert timedelta into minutes
        self.target['STOP_TIME'] = self.target.apply(lambda x: parse_minutes(x['ALL_PLAINS_TIME']), axis=1)
        print(self.target['STOP_TIME'].sum())

        # Уберем из значений промежутки в которые была остановка производственного процесса
        for idx, row in self.target.iterrows():
            if row['STOP_TIME'] == 0:
                print('Empty:', row['START_DATE'], " --> ", row['END_DATE'])
                continue
            else:
                print('Non Empty:', row['START_DATE'], " --> ", row['END_DATE'])
                self.values = self.values.drop(self.values[(self.values["Iter_Data"] > row['START_DATE']) &
                                                           (self.values["Iter_Data"] < row['END_DATE'])
                                                           ].index)

                self.smoothed_values = self.smoothed_values.drop(
                    self.smoothed_values[(self.smoothed_values["Iter_Data"] > row['START_DATE']) &
                                         (self.smoothed_values["Iter_Data"] < row['END_DATE'])
                                         ].index)

    def smooth_control_results(self):
        self.smoothed_values = copy.deepcopy(self.values)
        for cont_res in self.values:
            self.smoothed_values[cont_res] = savgol_filter(self.values[cont_res], 15, 3)

        self.smoothed_values['Iter_Data'] = pd.to_datetime(self.smoothed_values['Iter_Data'])

    def load_target_variable(self, fiheadlepath):
        # TODO here should be loaded from target variables and checked them
        return 0

    def mark_target(self, left_delta):
        assert isinstance(left_delta, int)
        assert isinstance(self.target, pd.DataFrame)
        self.values["target_variable"] = 0
        self.smoothed_values["target_variable"] = 0
        for idx, row in self.target.iterrows():
            print(idx, ' -----> ', )
            left_delta = pd.to_timedelta(left_delta, 'm')
            filded_condition = ((self.values.index.to_pydatetime() <= row['START_DATE'])
                                & (self.values.index.to_pydatetime() >= row['START_DATE'] - left_delta))

            self.values.loc[filded_condition, 'target_variable'] = 1
            self.smoothed_values.loc[filded_condition, 'target_variable'] = 1

        print(self.values.target_variable.value_counts())

    # Only with 30 time delta
    def mark_self_target(self, left_delta, impulse_delta):
        output = list()
        assert isinstance(left_delta, int)
        assert isinstance(self.target, pd.DataFrame)
        self.values["target_variable"] = 0
        self.smoothed_values["target_variable"] = 0
        non_anomaly_set = signals.values.iloc[(signals.values['target_variable'] != 1).values]
        for idx, row in self.target.iterrows():
            print(idx, ' -----> ', )
            left_delta = pd.to_timedelta(left_delta, 'm')
            filded_condition = ((self.values.index.to_pydatetime() <= row['START_DATE'])
                                & (self.values.index.to_pydatetime() >= row['START_DATE'] - left_delta))

            self.values.loc[filded_condition, 'target_variable'] = 1
            self.smoothed_values.loc[filded_condition, 'target_variable'] = 1
            print('--->')
            condition = (self.values.index.to_pydatetime() <= row['START_DATE'])

            vl = self.values.iloc[condition].drop(columns=["Iter_Data"]).values[:impulse_delta]
            vl = np.array(vl)
            output.append(vl)
            print(vl.shape)

            clean_condition = (non_anomaly_set.index.to_pydatetime() <= row['START_DATE'])
            clear_target = non_anomaly_set.iloc[clean_condition].drop(columns=["Iter_Data"]).values[:impulse_delta]
            clear_target = np.array(clear_target)
            print(clear_target.shape)
            output.append(clear_target)

        # non_anomaly_set = signals.values.iloc[(signals.values['target_variable'] != 1).values]
        # for idx, row in self.target.iterrows():
        #     clean_condition = (non_anomaly_set.index.to_pydatetime() <= row['START_DATE'])
        #     clear_target = non_anomaly_set.iloc[clean_condition].drop(columns=["Iter_Data"]).values[:46]
        #     clear_target = np.array(clear_target)
        #     print(clear_target.shape)
        #     output.append(clear_target)

        # print(self.values.target_variable.value_counts())
        # output = [item for item in output if item.shape[0] == 46]
        self.special_marking_data = np.vstack(output)
        print(self.special_marking_data.shape)
        # Only with 30 time delta
        # self.special_marking_data = self.special_marking_data.reshape(77, impulse_delta, 41)

    #     def stainar Ффункция по избавлению от автокорреляции
    """ Display Functions """

    # TODO operation mode (jupyter and standart)
    def display_values(self):
        for column in self.values.columns[1:]:
            plt.figure(figsize=(15, 3))
            plt.title(column)
            plt.plot(self.values[column].values[:50000])
            plt.show()

    def display_smoothed_values(self):
        for column in self.smoothed_values.columns[1:]:
            plt.figure(figsize=(15  , 3))
            plt.title(column)
            plt.plot(self.smoothed_values[column].values[:50000])
            plt.show()


if __name__ == '__main__':
    signals = Signal("datasets/DATA/Nikol/L1_10-12.18.csv")
    signals.smooth_control_results()
    # Read Target Value
    signals.read_target()
    # Clear and preproccessing target value
    signals.clear_target()
    # Mark up target value into signals.values
    signals.mark_self_target(left_delta=30, impulse_delta=60)
    print(signals.special_marking_data)