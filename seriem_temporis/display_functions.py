import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def display_block_of_values(dataframe=pd.DataFrame,
                            x_label='x_label',
                            y_label='y_label',
                            figsize=(15, 30),
                            plot_size=(30, 2)):
    """

    :param dataframe:
    :param x_label:
    :param y_label:
    :param figsize:
    :param plot_size:
    :return:
    """
    assert isinstance(dataframe, pd.DataFrame)
    plt.figure(figsize=figsize)
    for index, col in enumerate(dataframe.columns):
        plt.subplot(plot_size[0], plot_size[1], index+1)
        plt.plot(dataframe[col])
        plt.title(col)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history['accuracy'])+1), model_history.history['accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# Функция отображения историия обучения модели
def plot_model_history_2(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
