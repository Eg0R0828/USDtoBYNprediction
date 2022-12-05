# -*- coding: utf-8 *-*


import pandas, tensorflow, numpy
import matplotlib.pyplot as plt


def test_saved_nn_model(n, test_data_file_path='./testData.csv'):
    try:
        nn_model = tensorflow.keras.models.load_model('./nn_model.nn')
    except BaseException:
        print('ERROR: Saved NN-model file was deleted or damaged!')
        return

    try:
        data_frame = pandas.read_csv(test_data_file_path, header=None)
    except BaseException:
        print('ERROR: Wrong test data set file path!')
        return

    data_set = data_frame.values
    data = data_set.astype(float)

    timeline = [i - 1 for i in range(1, data.shape[0]) if i % n == 0]
    right_data = numpy.zeros(len(timeline), float)
    for i in range(right_data.shape[0]):
        right_data[i] = data[i * n + n - 1][0]

    input_data = numpy.ndarray((len(timeline), 3))
    for i in range(input_data.shape[0]):
        input_data[i][0] = data[i * n][0]
        input_data[i][1] = data[i * n][0]; input_data[i][2] = data[i * n][0]
        for j in range(n):
            if data[i * n + j][0] < input_data[i][1]:
                input_data[i][1] = data[i * n + j][0]
            if data[i * n + j][0] > input_data[i][2]:
                input_data[i][2] = data[i * n + j][0]
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    predicted_data = nn_model.predict(input_data)

    plt.plot(timeline, right_data, 'b', label='Right values')
    plt.plot(timeline, predicted_data, 'y', label='Predicted values')
    plt.title('USD to BYN values (2022 year)')
    plt.xlabel('Timeline')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_saved_nn_model(7)
