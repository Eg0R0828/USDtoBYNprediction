# -*- coding: utf-8 *-*


import pandas, tensorflow, numpy
import matplotlib.pyplot as plt


def create_nn_model(n):
    # Reading data set
    try:
        data_frame = pandas.read_csv('./dataSet.csv', header=None)
    except BaseException:
        print('ERROR: Training data set file was deleted or damaged!')
        return
    data_set = data_frame.values
    data = data_set.astype(float)

    # Preparing training input data (X) and training right answers (Y): X - [0]input; [1]min; [2]max; Y - output
    X = numpy.ndarray((int(len(data) / n), 3))
    Y = numpy.ndarray((X.shape[0], 1))
    for i in range(X.shape[0]):
        X[i][0] = data[i * n][0]
        Y[i][0] = data[i * n + n - 1][0]
        X[i][1] = X[i][0]; X[i][2] = X[i][0]
        for j in range(n):
            if data[i * n + j][0] < X[i][1]:
                X[i][1] = data[i * n + j][0]
            if data[i * n + j][0] > X[i][2]:
                X[i][2] = data[i * n + j][0]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Preparing model (setting a model type; input, hidden and output neuron levels)
    nn_model = tensorflow.keras.models.Sequential()
    nn_model.add(tensorflow.keras.layers.LSTM(X.shape[1], return_sequences=True, input_shape=(X.shape[1], 1)))
    nn_model.add(tensorflow.keras.layers.LSTM(X.shape[1], return_sequences=False))
    nn_model.add(tensorflow.keras.layers.Dense(1))

    # Compiling and training model
    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    training_history = nn_model.fit(X, Y, epochs=5, batch_size=1, validation_split=0.1)
    nn_model.save('./nn_model.nn')

    # Getting training metrics data
    loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']
    acc = training_history.history['mean_absolute_error']
    val_acc = training_history.history['val_mean_absolute_error']
    epochs = range(1, len(loss) + 1)

    # Drawing training metrics graphics and printing training results
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training mae')
    plt.plot(epochs, val_acc, 'b', label='Validation mae')
    plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    results = nn_model.evaluate(X, Y)
    print(results)


if __name__ == '__main__':
    create_nn_model(7)
