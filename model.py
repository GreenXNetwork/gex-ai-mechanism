#!/usr/bin/env python3
##########################################################################################
# Author: Tung Kieu
# Date Started: 2018-04-07
# Purpose: Train recurrent neural network to classify Time Series.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################
import datetime
import numpy as np
import tensorflow as tf
from data import LoadDataWithoutRatio, WriteFile
import os

##########################################################################################
# Write file
##########################################################################################

log_file_name = './/log//TestLSTM_' + str(datetime.date.today()) + '.log'

##########################################################################################
# Load data
##########################################################################################


for root, dirs, files in os.walk('UCR//1', topdown=False):
    for dataset in dirs:
        tf.reset_default_graph()
        x_train, x_test, yTrain_enc, yTest_enc, max_sequence_length, number_of_class = LoadDataWithoutRatio('UCR//1', dataset)
        print('--------------------------------------------------')
        print(dataset)
        WriteFile(log_file_name, 'a', dataset)

        ##########################################################################################
        # Settings
        ##########################################################################################

        # Model settings
        #
        num_features = 1
        num_steps = max_sequence_length
        num_cells = 64
        num_classes = number_of_class

        # Training parameters
        #
        epochs = 4000
        batch_size = 64
        learning_rate = 1e-3

        ##########################################################################################
        # Operators
        ##########################################################################################

        # Inputs
        #
        x = tf.placeholder(tf.float32, [None, num_steps, num_features])
        y = tf.placeholder(tf.float32, [None, num_classes])

        # Variables
        #
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_cells)  # Modified code to run RWA model
        W_end = tf.Variable(tf.truncated_normal([num_cells, num_classes], mean=0.0, stddev=0.1))
        b_end = tf.Variable(tf.zeros([num_classes]))

        # Model
        #
        with tf.variable_scope('layer_1'):
            h, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        with tf.variable_scope('layer_output'):
            ly = tf.matmul(h[:, num_steps - 1, :], W_end) + b_end
            py = tf.nn.softmax(ly)

        # Cost function and optimizer
        #
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ly, labels=y))  # Cross-entropy cost function.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # Evaluate performance
        #
        correct = tf.equal(tf.argmax(py, 1), tf.argmax(y, 1))
        accuracy = 100.0 * tf.reduce_mean(tf.cast(correct, tf.float32))

        # Create operator to initialize session
        #
        initializer = tf.global_variables_initializer()

        # Create operator for saving the model and its parameters
        #
        saver = tf.train.Saver()

        ##########################################################################################
        # Session
        ##########################################################################################
        # Start session
        init = tf.global_variables_initializer()

        print('Start the computation graph')
        with tf.Session() as sess:
            sess.run(init)
            print('Initialized')
            # Open session
            #
            if batch_size > x_train.shape[0]:
                feed_dict_train = {x: x_train, y: yTrain_enc}
                feed_dict_test = {x: x_test, y: yTest_enc}
                for epoch in range(epochs):
                    _tr, l_tr = sess.run([optimizer, cost], feed_dict=feed_dict_train)

                    if epoch % 100 == 0:
                        # Print report to user
                        #
                        print('  Iteration:', epoch)
                        print('  Cost (Training):      ', cost.eval(feed_dict_train), 'bits')
                        print('  Accuracy (Training):  ', accuracy.eval(feed_dict_train), '%')
                        print('  Cost (Test):    ', cost.eval(feed_dict_test), 'bits')
                        print('  Accuracy (Test):', accuracy.eval(feed_dict_test), '%')
                        print('', flush=True)

                # Save the trained model
                #
                sess.run([optimizer, cost], feed_dict={x: x_test, y: yTest_enc})
                print('--------------------------------------------------')
                print('Testing accuracy: {:.1f}'.format(accuracy.eval({x: x_test, y: yTest_enc})))
                print('--------------------------------------------------')
                WriteFile(log_file_name, 'a', 'Testing accuracy: {:.1f}'.format(accuracy.eval({x: x_test, y: yTest_enc})))
                saver.save(sess, 'bin/TestRLSTM_' + dataset + '.ckpt')
            else:
                for epoch in range(epochs):
                    offset = (epoch * batch_size) % (x_train.shape[0] - batch_size)
                    batch_data_train = x_train[offset:(offset + batch_size), :]
                    batch_labels_train = yTrain_enc[offset:(offset + batch_size), :]
                    feed_dict_train = {x: batch_data_train, y: batch_labels_train}
                    batch_data_test = x_test[offset:(offset + batch_size), :]
                    batch_labels_test = yTest_enc[offset:(offset + batch_size), :]
                    feed_dict_test = {x: batch_data_test, y: batch_labels_test}

                    _tr, l_tr = sess.run([optimizer, cost], feed_dict=feed_dict_train)

                    if epoch % 100 == 0:
                        # Print report to user
                        #
                        print('  Iteration:', epoch)
                        print('  Cost (Training):      ', cost.eval(feed_dict_train), 'bits')
                        print('  Accuracy (Training):  ', accuracy.eval(feed_dict_train), '%')
                        print('  Cost (Test):    ', cost.eval(feed_dict_test), 'bits')
                        print('  Accuracy (Test):', accuracy.eval(feed_dict_test), '%')
                        print('', flush=True)

                # Save the trained model
                #
                sess.run([optimizer, cost], feed_dict={x: x_test, y: yTest_enc})
                print('--------------------------------------------------')
                print('Testing accuracy: {:.1f}'.format(accuracy.eval({x: x_test, y: yTest_enc})))
                print('--------------------------------------------------')
                WriteFile(log_file_name, 'a', 'Testing accuracy: {:.1f}'.format(accuracy.eval({x: x_test, y: yTest_enc})))
                saver.save(sess, 'bin/TestLSTM_' + dataset + '.ckpt')