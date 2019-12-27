import numpy as np
import pandas as pd
import copy
from scipy.signal import savgol_filter


class SOP:
    control_results = None
    smoothed_control_results = None
    hord_shift = 145
    pc_shift = 50
    pc_schemes = [3, 4, 5, 6, 9, 10]
    hord_schemes = [1, 2, 7, 8, 11, 12, 13, 14, 15, 16]
    transverse_defects_columns = [7, 8, 15, 16]
    longitudinal_defects_set = None  # Список тактов отслеживающих продольные дефекты
    transverse_defects_set = None  # Список тактов отслеживающих поперечные дефекты

    def __init__(self, filepath="SOP1.txt"):
        self.filepath = filepath

    def read_control_results(self):
        # В рамках текущего модуля пока избавимся от первого столбца с номерами измерений, пока в них нет необходимости
        splited_info = ""
        with open(self.filepath, mode="r", encoding="cp1251") as f:
            splited_info = f.readline()
        self.control_results = pd.read_csv(self.filepath, encoding="cp1251", delimiter=" ", header=None, skiprows=1)
        self.control_results = self.control_results.drop(columns=[0])
        print(self.control_results.shape[0])
        splited_info = splited_info.split(' ')
        self.date = splited_info[1]
        self.time = splited_info[2]
        self.temperature = splited_info[3]

    def fix_shift_issue(self):
        self.control_results = self.control_results.iloc[:1024]
        self.smoothed_control_results = self.smoothed_control_results.iloc[:1024]
        for scheme in self.pc_schemes:
            self.control_results[scheme] = np.roll(self.control_results[scheme], -self.pc_shift)
            self.smoothed_control_results[scheme] = np.roll(self.smoothed_control_results[scheme], -self.pc_shift)

        for scheme in self.hord_schemes:
            self.control_results[scheme] = np.roll(self.control_results[scheme], -self.hord_shift)
            self.smoothed_control_results[scheme] = np.roll(self.smoothed_control_results[scheme], -self.hord_shift)

    def smooth_control_results(self):
        self.smoothed_control_results = copy.deepcopy(self.control_results)
        for cont_res in self.control_results:
            self.smoothed_control_results[cont_res] = savgol_filter(self.control_results[cont_res], 15, 3)

    def split_by_defects(self):
        self.longitudinal_defects_set = self.smoothed_control_results.drop(self.transverse_defects_columns, axis=1)
        self.transverse_defects_set = self.smoothed_control_results[self.transverse_defects_columns]

    def load_target_variable(self, filepath):
        target = pd.read_csv(filepath)
        self.longitudinal_defects_set['y'] = 0
        self.transverse_defects_set['y_t'] = 0
        # Fill l defect
        for index, row in target[target['Тип дефекта'] == "L"].iterrows():
            self.longitudinal_defects_set["y"][
            int(row["Начало дефекта"]):int(row["Начало дефекта"]) + int(row["Длина дефекта"])] = 1

        for index, row in target[target['Тип дефекта'] == "T"].iterrows():
            self.transverse_defects_set["y_t"][int(row["Начало дефекта"]):15 + int(row["Начало дефекта"])] = 1
