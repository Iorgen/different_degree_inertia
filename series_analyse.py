from seriem_temporis.signal_manipulation import AnomaliesLibrary


if __name__ == "__main__":
    anomaly_functions = [AnomaliesLibrary.change_trend,
                         AnomaliesLibrary.decrease_dispersion,
                         AnomaliesLibrary.dome,
                         AnomaliesLibrary.increase_dispersion,
                         AnomaliesLibrary.shift_trend]
    handler_conf = {
        "name": "Testing Model",

        # Anomaly generator configs
        "generate_anomaly_mode": True,
        "anomaly_functions": anomaly_functions,
        "rolling_window_size": 500,
        "minimal_anomaly_length": 50,
        "sample_rate": 40,

        # reader configs
        "folder_path": "/media/jurgen/Новый том/Sygnaldatasets/kaspersky/attacks/",
        "delimiter": ",",
        "encoding": "cp1251",

        # Without anomaly generator
        "target_variable": None,
        "target_values": None,

        # preprocess configs
        "smooth_method": "savgol",
        "normalize": True,
        "scale": True,
        "smooth": True,
    }

    print('start')

    # neural_model = SplitConvolutionalAnomalyDetector()
    # neural_model.fit_model(verbose=1, epochs_per_step=5, batch_size=128)
