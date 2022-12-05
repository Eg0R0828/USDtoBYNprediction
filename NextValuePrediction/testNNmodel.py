# -*- coding: utf-8 *-*


import pandas, tensorflow


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
    data = data.reshape(1, data.shape[0], data.shape[1])

    print(nn_model.predict(data[:, 0:n, :]))


if __name__ == '__main__':
    test_saved_nn_model(50)
