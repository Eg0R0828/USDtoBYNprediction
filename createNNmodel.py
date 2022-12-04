# -*- coding: utf-8 *-*


import pandas, tensorflow, numpy
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Reading data set
    data_frame = pandas.read_csv('./dataSet.csv', header=None)
    data_set = data_frame.values
    data = data_set.astype(float)

    # Count of previous values for predicting
    N = 50

    # Preparing training input data (X) and training right answers (Y)
    X = numpy.ndarray((len(data) - N, N))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = data[j + i][0]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    Y = numpy.ndarray((len(data) - N, 1))
    for i in range(Y.shape[0]):
        Y[i][0] = data[i + N][0]

    # Preparing model (setting a model type; input, hidden and output neuron levels)
    nn_model = tensorflow.keras.models.Sequential()
    nn_model.add(tensorflow.keras.layers.LSTM(N, return_sequences=True, input_shape=(N, 1)))
    nn_model.add(tensorflow.keras.layers.LSTM(N, return_sequences=False))
    nn_model.add(tensorflow.keras.layers.Dense(1))

    # Compiling and training model
    nn_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    training_history = nn_model.fit(X, Y, epochs=10, batch_size=1, validation_split=0.1)
    nn_model.save('./nn_model.nn')

    # Getting training metrics data
    loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']
    acc = training_history.history['acc']
    val_acc = training_history.history['val_acc']
    epochs = range(1, len(loss) + 1)

    # Drawing training metrics graphics and printing training results
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # построение графика точности
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    results = nn_model.evaluate(X, Y)
    print(results)
