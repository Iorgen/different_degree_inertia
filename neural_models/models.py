import copy
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from .generators import dataset_reader


class SplitConvolutionalAnomalyDetector:

    # Model training configuration
    EPOCHS = 200
    BATCH_SIZE = 32
    PATIENCE = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0005
    LR_DECAY = 0.0001

    # Retraining configuration
    TRAINABLE = False
    WEIGHTS = "model-0.27.h5"

    # Multithreading computing params
    MULTITHREADING = False
    THREADS = 1

    # Detection model
    MODEL = None
    trainX = None
    trainY = None

    def __init__(self):
        self.DATASET_READER = dataset_reader('/media/jurgen/Новый том/Sygnaldatasets/kaspersky/attacks/')
        for input_train, output_train in self.DATASET_READER:
            self.init_model(input_train, output_train)
            break

    def get_model(self):
        return copy.deepcopy(self.MODEL)

    def init_model(self, train_x, train_y):
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # head 1
        inputs1 = Input(shape=(n_timesteps, n_features))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = TimeDistributed(Flatten())(drop1)
        # head 2
        inputs2 = Input(shape=(n_timesteps, n_features))
        conv2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)

        flat2 = TimeDistributed(Flatten())(drop2)
        # head 3
        inputs3 = Input(shape=(n_timesteps, n_features))
        conv3 = Conv1D(filters=64, kernel_size=11, activation='relu', padding='same')(inputs3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = TimeDistributed(Flatten())(drop3)

        # Merge all convolutions in one
        merged = concatenate([flat1, flat2, flat3])
        dense1 = Dense(100, activation='relu')(merged)
        outputs = GRU(2, activation="softmax", return_sequences=True)(dense1)

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        model.compile(loss="mse", optimizer="Adam", metrics=['accuracy'])
        self.MODEL = model

    def fit_model(self, verbose=1, epochs_per_step=5, batch_size=128):
        # Anti over-fitting callback
        es = EarlyStopping(monitor='loss', mode='min')
        # Fit model using dynamically uploading dataset
        for input_train, output_train in self.DATASET_READER:
            self.MODEL.fit([input_train, input_train, input_train], output_train, epochs=epochs_per_step,
                           batch_size=batch_size, verbose=verbose, callbacks=[es])
