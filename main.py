# -*- coding: utf-8 *-*


from createNNmodel import create_nn_model
from testNNmodel import test_saved_nn_model


# A number of previous values for predicting
n = 50

if input('Do You want to create a new NN-model? (Yes / other word) ') == 'Yes':
    create_nn_model(n)
if input('Do You want to test saved NN-model? (Yes / other word) ') == 'Yes':
    test_saved_nn_model(n, input('Enter a test data file path: '))
