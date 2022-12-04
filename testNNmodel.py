# -*- coding: utf-8 *-*


import pandas, tensorflow, numpy
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


if __name__ == '__main__':
    nn_model = tensorflow.keras.models.load_model('./nn_model.nn')

    data_frame = pandas.read_csv('./testData.csv', header=None)
    data_set = data_frame.values
    data = data_set.astype(float)
    data = data.reshape(1, data.shape[0], data.shape[1])
    N = 50

    print(nn_model.predict(data[:, 0:N, :]))
