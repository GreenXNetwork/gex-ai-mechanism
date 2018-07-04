#!/usr/bin/env python3
##########################################################################################
# Author: Tung Kieu
# Date Started: 2018-04-07
# Purpose: Train recurrent neural network to classify Time Series.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################
import numpy as np

##########################################################################################
# Write log
##########################################################################################
def WriteFile(_file_name, _mode, _content):
    file = open(_file_name, _mode)
    file.writelines(_content + '\n')
    file.close()

##########################################################################################
# Load data
##########################################################################################
def LoadDataWithoutRatio(_direc, _dataset):
    '''
    Load dataset from UCR
    param _direc: name of entire folder (UCR)
    :param _dataset: name of dataset (e.g., Adiac)
    :return: x_train, x_test, yTrain_enc, yTest_enc, max_sequence_length, number_of_class
    '''
    data_dir = _direc + '/' + _dataset + '/' + _dataset
    data_train = np.loadtxt(data_dir + '_TRAIN', delimiter=',')
    data_test = np.loadtxt(data_dir + '_TEST', delimiter=',')
    # Extract x_train and x_test
    x_train = data_train[:, 1:]
    x_test = data_test[:, 1:]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    # Calculate max sequence length
    length_x_train = x_train.shape[1]
    length_x_test = x_test.shape[1]
    max_sequence_length = max(length_x_train, length_x_test)
    # Extract y_train and y_test
    y_train = data_train[:, 0].astype(int)
    y_test = data_test[:, 0].astype(int)
    # Move y from [1:n] -> [0:n-1]
    y_train = y_train - 1
    y_test = y_test - 1
    # Calculate number of class
    number_of_class_train = np.unique(y_train).shape[0]
    number_of_class_test = np.unique(y_test).shape[0]
    number_of_class = max(number_of_class_train, number_of_class_test)
    yTrain_enc = np.zeros((y_train.shape[0], number_of_class), dtype='int32')
    # empty one-hot matrix
    yTrain_enc[np.arange(y_train.shape[0]), y_train] = 1
    yTest_enc = np.zeros((y_test.shape[0], number_of_class), dtype='int32')
    # empty one-hot matrix
    yTest_enc[np.arange(y_test.shape[0]), y_test] = 1
    return x_train, x_test, yTrain_enc, yTest_enc, max_sequence_length, number_of_class