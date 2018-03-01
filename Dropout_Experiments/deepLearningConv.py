from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import argparse
import sys
import time
import csv

from sys import byteorder
from numpy import size

from tensorflow.python.framework import dtypes

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot

FLAGS = None


# Initialize the DataSets for the permuted MNIST task
def initDataSetsPermutation():
    # Variable to read out the labels & data
    mnistData = read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

    # MNIST labels & data for training
    mnistLabelsTrain = mnistData.train.labels
    mnistDataTrain = mnistData.train.images

    # MNIST labels & data for testing
    mnistLabelsTest = mnistData.test.labels
    mnistDataTest = mnistData.test.images

    mnistPermutationTrain = np.array(mnistDataTrain, dtype=np.float32)
    mnistPermutationTest = np.array(mnistDataTest, dtype=np.float32)

    # Concatenate both arrays to make sure the shuffling is consistent over
    # the training and testing sets, split them afterwards
    mnistPermutation = np.concatenate([mnistDataTrain, mnistDataTest])
    np.random.shuffle(mnistPermutation.T)
    mnistPermutationTrain, mnistPermutationTest = np.split(mnistPermutation, [
        mnistDataTrain.shape[0]])

    # Create new data set objects out of permuted MNIST images
    global dataSetOneTrain
    dataSetOneTrain = DataSet(255. * mnistDataTrain,
                              mnistLabelsTrain, reshape=False)
    global dataSetOneTest
    dataSetOneTest = DataSet(255. * mnistDataTest,
                             mnistLabelsTest, reshape=False)

    global dataSetTwoTrain
    dataSetTwoTrain = DataSet(255. * mnistPermutationTrain,
                              mnistLabelsTrain, reshape=False)
    global dataSetTwoTest
    dataSetTwoTest = DataSet(255. * mnistPermutationTest,
                             mnistLabelsTest, reshape=False)


# Initialize the DataSets for the partitioned digits task
def initDataSetsExcludedDigits():
    args = parser.parse_args()
    if args.exclude[0:]:
        labelsToErase = [int(i) for i in args.exclude[0:]]

    # Variable to read out the labels & data.
    mnistData = read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=False)

    # MNIST labels & data for training
    mnistLabelsTrain = mnistData.train.labels
    mnistDataTrain = mnistData.train.images

    # MNIST labels & data for testing
    mnistLabelsTest = mnistData.test.labels
    mnistDataTest = mnistData.test.images

    # Filtered labels & data for training (DataSetOne).
    labelsExcludedTrain = np.array([mnistLabelsTrain[i] for i in xrange(0,
                                                                        mnistLabelsTrain.shape[0]) if
                                    mnistLabelsTrain[i]
                                    in labelsToErase], dtype=np.uint8)

    dataExcludedTrain = np.array([mnistDataTrain[i, :] for i in xrange(0,
                                                                       mnistLabelsTrain.shape[0]) if mnistLabelsTrain[i]
                                  in labelsToErase], dtype=np.float32)

    # Filtered labels & data for testing (DataSetOne).
    labelsExcludedTest = np.array([mnistLabelsTest[i] for i in xrange(0,
                                                                      mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                                   in labelsToErase], dtype=np.uint8)

    dataExcludedTest = np.array([mnistDataTest[i, :] for i in xrange(0,
                                                                     mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                                 in labelsToErase], dtype=np.float32)

    # Filtered labels & data for training (DataSetTwo).
    labelsKeepedTrain = np.array([mnistLabelsTrain[i] for i in xrange(0,
                                                                      mnistLabelsTrain.shape[0]) if mnistLabelsTrain[i]
                                  not in labelsToErase], dtype=np.uint8)

    dataKeepedTrain = np.array([mnistDataTrain[i, :] for i in xrange(0,
                                                                     mnistLabelsTrain.shape[0]) if mnistLabelsTrain[i]
                                not in labelsToErase], dtype=np.float32)

    # Filtered labels & data for testing (DataSetTwo).
    labelsKeepedTest = np.array([mnistLabelsTest[i] for i in xrange(0,
                                                                    mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                                 not in labelsToErase], dtype=np.uint8)

    dataKeepedTest = np.array([mnistDataTest[i, :] for i in xrange(0,
                                                                   mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                               not in labelsToErase], dtype=np.float32)

    # Transform labels to one-hot vectors
    labelsKeepedTrainOnehot = dense_to_one_hot(labelsKeepedTrain, 10)
    labelsExcludedTrainOnehot = dense_to_one_hot(labelsExcludedTrain, 10)

    labelsKeepedTestOnehot = dense_to_one_hot(labelsKeepedTest, 10)
    labelsExcludedTestOnehot = dense_to_one_hot(labelsExcludedTest, 10)

    labelsAllTrainOnehot = dense_to_one_hot(mnistLabelsTrain, 10)
    labelsAllTestOnehot = dense_to_one_hot(mnistLabelsTest, 10)

    # Create DataSets (1 -> digits we kept, 2 -> excluded digits, All -> all digits)
    global dataSetOneTrain
    dataSetOneTrain = DataSet(255. * dataKeepedTrain,
                              labelsKeepedTrainOnehot, reshape=False)
    global dataSetOneTest
    dataSetOneTest = DataSet(255. * dataKeepedTest,
                             labelsKeepedTestOnehot, reshape=False)

    global dataSetTwoTrain
    dataSetTwoTrain = DataSet(255. * dataExcludedTrain,
                              labelsExcludedTrainOnehot, reshape=False)
    global dataSetTwoTest
    dataSetTwoTest = DataSet(255. * dataExcludedTest,
                             labelsExcludedTestOnehot, reshape=False)

    global dataSetAllTrain
    dataSetAllTrain = DataSet(255. * mnistDataTrain,
                              labelsAllTrainOnehot, reshape=False)
    global dataSetAllTest
    dataSetAllTest = DataSet(255. * mnistDataTest,
                             labelsAllTestOnehot, reshape=False)


def train():
    # feed dictionary for dataSetOne
    def feed_dict_1(train):
        if train:
            xs, ys = dataSetOneTrain.next_batch(FLAGS.batch_size_1)
            k = FLAGS.dropout
        else:
            xs, ys = dataSetOneTest.images, dataSetOneTest.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # feed dictionary for dataSetTwo
    def feed_dict_2(train):
        if train:
            xs, ys = dataSetTwoTrain.next_batch(FLAGS.batch_size_2)
            k = FLAGS.dropout
        else:
            xs, ys = dataSetTwoTest.images, dataSetTwoTest.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # feed dictionary for dataSetAll
    def feed_dict_all(train):
        if train:
            xs, ys = dataSetAllTrain.next_batch(FLAGS.batch_size_all)
            k = FLAGS.dropout
        else:
            xs, ys = dataSetAllTest.images, dataSetAllTest.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # weights initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # biases initialization
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # define a 2d convolutional layer
    def conv_layer(input, channels_in, channels_out, name='conv'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable([5, 5, channels_in, channels_out])
            with tf.name_scope('biases'):
                b = bias_variable([channels_out])
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
            act = tf.nn.relu(conv + b)
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", act)
            return act

    # define a fully connected layer
    def fc_layer(input, channels_in, channels_out, name='fc'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable([channels_in, channels_out])
            with tf.name_scope('biases'):
                b = bias_variable([channels_out])
            act = tf.nn.relu(tf.matmul(input, W) + b)
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", act)
            return act

    # define a readout layer
    def ro_layer(input, channels_in, channels_out, name='read'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable([channels_in, channels_out])
            with tf.name_scope('biases'):
                b = bias_variable([channels_out])
            act = tf.matmul(input, W) + b
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", act)
            return act

    # pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # Start an interactive session
    sess = tf.InteractiveSession()

    # Placeholder variables for input data
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    # reshape x to a 4d tensor, 2nd & 3rd dimension = width & height, 4th = color
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 9)

    # Create the first convolutional layer
    # convolve x_image with the weight tensor, add the bias, apply ReLu
    h_conv1 = conv_layer(x_image, 1, 32, 'h_conv1')
    # max pooling for first conv layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Create the second convolutional layer
    h_conv2 = conv_layer(h_pool1, 32, 64, 'h_conv2')
    # max pooling for second conv layer
    h_pool2 = max_pool_2x2(h_conv2)

    # reshape tensor from the pooling layer into a batch of vectors
    h_pool2_flattened = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Create a densely Connected Layer
    # image size reduced to 7x7, add a fully-connected layer with 1024 neurons
    # to allow processing on the entire image.
    h_fc1 = fc_layer(h_pool2_flattened, 7 * 7 * 64, 1024, 'h_fc1')

    # Apply dropout to the densely connected layer
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Create a readout layer
    logits = ro_layer(h_fc1_drop, 1024, 10, 'ro_layer')

    # Define a softmax cross entropy loss model
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Use the ADAM optimizer to run the training with an initial learning rate
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)

    # Compute accuracy and correct predictions
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries and write them out to /tmp/tensorflow/mnist/logs (by default),
    # writers are used to log testing and training accuracy for the data-sets
    merged = tf.summary.merge_all()

    train_writer_ds1 = tf.summary.FileWriter(FLAGS.log_dir + '/training_ds1', sess.graph)
    test_writer_ds1 = tf.summary.FileWriter(FLAGS.log_dir + '/testing_ds1')

    train_writer_ds2 = tf.summary.FileWriter(FLAGS.log_dir + '/training_ds2', sess.graph)
    test_writer_ds2 = tf.summary.FileWriter(FLAGS.log_dir + '/testing_ds2')

    test_writer_ds1_cf = tf.summary.FileWriter(FLAGS.log_dir + '/testing_ds1_cf')
    test_writer_dsc_cf = tf.summary.FileWriter(FLAGS.log_dir + '/testing_dsc_cf')

    # initialize variables & print information about the data sets.
    tf.global_variables_initializer().run()

    print('Deep Convolutional Neural Network')
    print('Files being logged to... %s' % (FLAGS.log_dir,))
    print('\nHyperparameters:')
    print('____________________________________________________________')
    print('\nTraining steps for first training (data set 1): %s' % (FLAGS.max_steps_ds1,))
    print('Training steps for second training (data set 2): %s' % (FLAGS.max_steps_ds2,))
    print('Batch size for data set 1: %s' % (FLAGS.batch_size_1,))
    print('Batch size for data set 2: %s' % (FLAGS.batch_size_2,))
    print('Dropout (keep_prob): %s' % (FLAGS.dropout,))
    print('Learning rate: %s' % (FLAGS.learning_rate,))
    print('\nInformation about the data sets:')
    print('____________________________________________________________')
    if FLAGS.exclude:
        print('\nExcluded digits: ')
        print('DataSetOne (train) contains: %s images.' % (dataSetOneTrain.labels.shape[0],))
        print('DataSetOne (test) contains: %s images.\n' % (dataSetOneTest.labels.shape[0],))
        print('DataSetTwo (train) contains: %s images.' % (dataSetTwoTrain.labels.shape[0],))
        print('DataSetTwo (test) contains: %s images.\n' % (dataSetTwoTest.labels.shape[0],))
        print('Original MNIST data-set (train) contains: %s images.' % (dataSetAllTrain.labels.shape[0],))
        print('Original MNIST data-set (test) contains: %s images.' % (dataSetAllTest.labels.shape[0],))
    if FLAGS.permutation:
        print('Permuted digits: ')
        print('DataSetOne (train) contains: %s images.' % (dataSetOneTrain.labels.shape[0],))
        print('DataSetOne (test) contains: %s images.\n' % (dataSetOneTest.labels.shape[0],))
        print('DataSetTwo (train) contains: %s images.' % (dataSetTwoTrain.labels.shape[0],))
        print('DataSetTwo (test) contains: %s images.\n' % (dataSetTwoTest.labels.shape[0],))

    print('\nTraining on DataSetOne...')
    print('____________________________________________________________')
    print(time.strftime('%X %x %Z'))
    # Training on DataSetOne
    for i in range(FLAGS.max_steps_ds1):
        if i % 5 == 0:  # Record test-set accuracy every 5 steps
            s, acc = sess.run([merged, accuracy], feed_dict=feed_dict_1(False))
            test_writer_ds1.add_summary(s, i)
            print('test set 1 accuracy at step: %s\t%s' % (i, acc))
        else:  # Run training steps and record training-set accuracy
            s, _ = sess.run([merged, train_step], feed_dict_1(True))
            train_writer_ds1.add_summary(s, i)
    train_writer_ds1.close()
    test_writer_ds1.close()

    print('\nTraining on DataSetTwo...')
    print('____________________________________________________________')
    print(time.strftime('%X %x %Z'))
    writer = csv.writer(open(FLAGS.plot_file, "wb"))

    # Training on DataSetTwo, logging every step to investigate CF
    for i in range(FLAGS.max_steps_ds2):
        # if i % 5 == 0: # Record test-set accuracy every 5 steps
        s1, acc1 = sess.run([merged, accuracy], feed_dict=feed_dict_2(False))
        test_writer_ds2.add_summary(s1, i)
        s2, acc2 = sess.run([merged, accuracy], feed_dict=feed_dict_1(False))
        test_writer_ds1_cf.add_summary(s2, i)
        if FLAGS.exclude:
            s3, accC = sess.run([merged, accuracy], feed_dict=feed_dict_all(False))
            test_writer_dsc_cf.add_summary(s3, i)
        print('test set 2 accuracy at step:%s\t%s' % (i, acc1))
        print('test set 1 accuracy at step:%s\t%s' % (i, acc2))
        if FLAGS.exclude:
            print('complete test set accuracy at step:%s\t%s' % (i, accC))
        # else: # Run training steps and record training-set accuracy
        s, _ = sess.run([merged, train_step], feed_dict_2(True))
        train_writer_ds2.add_summary(s, i)
        writer.writerow([i, acc2])
    train_writer_ds2.close()
    test_writer_ds2.close()
    test_writer_ds1_cf.close()
    if FLAGS.exclude:
        test_writer_dsc_cf.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if FLAGS.permutation:
        initDataSetsPermutation()
    if FLAGS.exclude:
        initDataSetsExcludedDigits()
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', type=int, nargs='*',
                        help="Exclude specified classes from the MNIST DataSet")
    parser.add_argument('--permutation', action='store_true',
                        help='Use a random but consistent permutation of the MNIST data set.')
    parser.add_argument('--max_steps_ds1', type=int, default=2000,
                        help='Number of steps to run trainer for data set 1.')
    parser.add_argument('--max_steps_ds2', type=int, default=100,
                        help='Number of steps to run trainer for data set 2.')
    parser.add_argument('--batch_size_1', type=int, default=100,
                        help='Size of the mini-batches we feed from dataSetOne.')
    parser.add_argument('--batch_size_2', type=int, default=100,
                        help='Size of the mini-batches we feed from dataSetTwo.')
    parser.add_argument('--batch_size_all', type=int, default=100,
                        help='Size of the mini-batches we feed from complete data set.')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Keep probability for training dropout.')
    parser.add_argument('--plot_file', type=str,
                        default='deepLearningConv.csv',
                        help='Filename for csv file to plot. Give .csv extension after file name.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs',
                        help='Summaries log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)