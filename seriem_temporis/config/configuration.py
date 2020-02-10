""" Class for preprocess file with signal information, anomaly generation, smoothing signal
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

handler_conf = {
    "name": None,

    # Anomaly generator configs
    "generate_anomaly_mode": None,
    "anomaly_functions": None,
    "rolling_window_size": None,
    "minimal_anomaly_length": None,
    "sample_rate": None,

    # reader configs
    "folder_path": None,
    "delimiter": None,
    "encoding": None,

    # Without anomaly generator
    "target_variable": None,
    "target_values": None,

    # preprocess configs
    "smooth_method": None,
    "normalize": None,
    "scale": None,
    "smooth": None,
}
