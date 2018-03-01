from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# implements LWTA catastrophic forgetting experiments of Srivastava et al. in Sec. 6


import tensorflow as tf
import numpy as np
import math
import os ;

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

# from hgext.histedit import action

FLAGS = None

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

wdict = {}


def initDataSetsClasses():
    global dataSetTrain
    global dataSetTest

    print(FLAGS.train_classes, FLAGS.test_classes)
    # Variable to read out the labels & data of the DataSet Object.
    mnistData = read_data_sets('./',
                               one_hot=True)
    
    # MNIST labels & data for training.
    mnistLabelsTrain = mnistData.train.labels
    mnistDataTrain = mnistData.train.images

    # MNIST labels & data for testing.
    mnistLabelsTest = mnistData.test.labels
    mnistDataTest = mnistData.test.images
    print("LABELS", mnistLabelsTest.shape, mnistLabelsTrain.shape)

    if FLAGS.permuteTrain != -1:
        # training dataset
        np.random.seed(FLAGS.permuteTrain)
        permTr = np.random.permutation(mnistDataTrain.shape[1])
        mnistDataTrainPerm = mnistDataTrain[:, permTr]
        mnistDataTrain = mnistDataTrainPerm;
        # dataSetTrain = DataSet(255. * dataSetTrainPerm,
        #                       mnistLabelsTrain, reshape=False)
    if FLAGS.permuteTest != -1:
        # testing dataset
        np.random.seed(FLAGS.permuteTest)
        permTs = np.random.permutation(mnistDataTest.shape[1])
        mnistDataTestPerm = mnistDataTest[:, permTs]
        # dataSetTest = DataSet(255. * dataSetTestPerm,
        #                      mnistLabelsTest, reshape=False)
        mnistDataTest = mnistDataTestPerm;

    if True:
        # args = parser.parse_args()
        print(FLAGS.train_classes, FLAGS.test_classes)
        if FLAGS.train_classes[0:]:
            labels_to_train = [int(i) for i in FLAGS.train_classes[0:]]

        if FLAGS.test_classes[0:]:
            labels_to_test = [int(i) for i in FLAGS.test_classes[0:]]

        # Filtered labels & data for training and testing.
        labels_train_classes = np.array([mnistLabelsTrain[i].argmax() for i in xrange(0,
                                                                                      mnistLabelsTrain.shape[0]) if
                                         mnistLabelsTrain[i].argmax()
                                         in labels_to_train], dtype=np.uint8)
        data_train_classes = np.array([mnistDataTrain[i, :] for i in xrange(0,
                                                                            mnistLabelsTrain.shape[0]) if
                                       mnistLabelsTrain[i].argmax()
                                       in labels_to_train], dtype=np.float32)

        labels_test_classes = np.array([mnistLabelsTest[i].argmax() for i in xrange(0,
                                                                                    mnistLabelsTest.shape[0]) if
                                        mnistLabelsTest[i].argmax()
                                        in labels_to_test], dtype=np.uint8)
        data_test_classes = np.array([mnistDataTest[i, :] for i in xrange(0,
                                                                          mnistDataTest.shape[0]) if
                                      mnistLabelsTest[i].argmax()
                                      in labels_to_test], dtype=np.float32)

        labelsTrainOnehot = dense_to_one_hot(labels_train_classes, 10)
        labelsTestOnehot = dense_to_one_hot(labels_test_classes, 10)

        dataSetTrain = DataSet(255. * data_train_classes,
                               labelsTrainOnehot, reshape=False)
        dataSetTest = DataSet(255. * data_test_classes,
                              labelsTestOnehot, reshape=False)


def train():
    args = parser.parse_args()
    training_readout_layer = args.training_readout_layer
    testing_readout_layer = args.testing_readout_layer
    LOG_FREQUENCY = args.test_frequency
    if not os.path.exists("./checkpoints"):
      os.mkdir("./checkpoints")


    # feed dictionary for dataSetOne
    def feed_dict(train, i):
        if train:
            xs, ys = dataSetTrain.next_batch(FLAGS.batch_size)
            k = FLAGS.dropout
        else:
            xs, ys = dataSetTest.images, dataSetTest.labels
            k = 1.0
        return {x: xs, y_: ys, global_step: i, keep_prob: k, }

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

    # Start an Interactive session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False;
    # sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)

    # Placeholder for input variables
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 9)

    global global_step
    global_step = tf.placeholder(tf.float32, shape=[], name="step");

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

    # Create a softmax linear classification layer for the outputs
    logits_tr1 = ro_layer(h_fc1_drop, 1024, 10, 'ro_layer_tr1')
    logits_tr2 = ro_layer(h_fc1_drop, 1024, 10, 'ro_layer_tr2')
    logits_tr3 = ro_layer(h_fc1_drop, 1024, 10, 'ro_layer_tr3')
    logits_tr4 = ro_layer(h_fc1_drop, 1024, 10, 'ro_layer_tr4')
    logits_trAll = logits_tr1 + logits_tr2 + logits_tr3 + logits_tr4 ;

    # Define the loss model as a cross entropy with softmax layer 1
    with tf.name_scope('cross_entropy_tr1'):
        diff_tr1 = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_tr1)
        with tf.name_scope('total_tr1'):
            cross_entropy_tr1 = tf.reduce_mean(diff_tr1)
    # tf.summary.scalar('cross_entropy_tr1', cross_entropy_tr1)

    # Define the loss model as a cross entropy with softmax layer 2
    with tf.name_scope('cross_entropy_tr2'):
        diff_tr2 = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_tr2)
        with tf.name_scope('total_tr2'):
            cross_entropy_tr2 = tf.reduce_mean(diff_tr2)
    # tf.summary.scalar('cross_entropy_tr2', cross_entropy_tr2)

    # Define the loss model as a cross entropy with softmax layer 3
    with tf.name_scope('cross_entropy_tr3'):
        diff_tr3 = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_tr3)
        with tf.name_scope('total_tr3'):
            cross_entropy_tr3 = tf.reduce_mean(diff_tr3)
    # tf.summary.scalar('cross_entropy_tr3', cross_entropy_tr3)

    # Define the loss model as a cross entropy with softmax layer 4
    with tf.name_scope('cross_entropy_tr4'):
        diff_tr4 = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits_tr4)
        with tf.name_scope('total_tr4'):
            cross_entropy_tr4 = tf.reduce_mean(diff_tr4)
    # tf.summary.scalar('cross_entropy_tr4', cross_entropy_tr4)

    # Use Gradient descent optimizer for training steps and minimize x-entropy
    # decaying learning rate tickles a few more 0.1% out of the algorithm
    with tf.name_scope('train_tr1'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        FLAGS.decayStep, FLAGS.decayFactor, staircase=True)

        train_step_tr1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_tr1)

    with tf.name_scope('train_tr2'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        FLAGS.decayStep, FLAGS.decayFactor, staircase=True)

        train_step_tr2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_tr2)

    with tf.name_scope('train_tr3'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        FLAGS.decayStep, FLAGS.decayFactor, staircase=True)

        train_step_tr3 = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_tr3)

    with tf.name_scope('train_tr4'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        FLAGS.decayStep, FLAGS.decayFactor, staircase=True)

        train_step_tr4 = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_tr4)

    # Compute correct prediction and accuracy
    with tf.name_scope('accuracy_tr1'):
        with tf.name_scope('correct_prediction_tr1'):
            correct_prediction_tr1 = tf.equal(tf.argmax(logits_tr1, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy_tr1'):
            accuracy_tr1 = tf.reduce_mean(tf.cast(correct_prediction_tr1, tf.float32))
    tf.summary.scalar('accuracy_tr1', accuracy_tr1)

    # Compute correct prediction and accuracy
    with tf.name_scope('accuracy_tr2'):
        with tf.name_scope('correct_prediction_tr2'):
            correct_prediction_tr2 = tf.equal(tf.argmax(logits_tr2, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy_tr2'):
            accuracy_tr2 = tf.reduce_mean(tf.cast(correct_prediction_tr2, tf.float32))
    tf.summary.scalar('accuracy_tr2', accuracy_tr2)

    # Compute correct prediction and accuracy
    with tf.name_scope('accuracy_tr3'):
        with tf.name_scope('correct_prediction_tr3'):
            correct_prediction_tr3 = tf.equal(tf.argmax(logits_tr3, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy_tr3'):
            accuracy_tr3 = tf.reduce_mean(tf.cast(correct_prediction_tr3, tf.float32))
    tf.summary.scalar('accuracy_tr3', accuracy_tr3)

    # Compute correct prediction and accuracy
    with tf.name_scope('accuracy_tr4'):
        with tf.name_scope('correct_prediction_tr4'):
            correct_prediction_tr4 = tf.equal(tf.argmax(logits_tr4, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy_tr4'):
            accuracy_tr4 = tf.reduce_mean(tf.cast(correct_prediction_tr4, tf.float32))
    tf.summary.scalar('accuracy_tr4', accuracy_tr4)
    
    # Compute correct prediction and accuracy
    with tf.name_scope('accuracy_trAll'):
        with tf.name_scope('correct_prediction_trAll'):
            correct_prediction_trAll = tf.equal(tf.argmax(logits_trAll, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy_trAll'):
            accuracy_trAll = tf.reduce_mean(tf.cast(correct_prediction_trAll, tf.float32))
    tf.summary.scalar('accuracy_trAll', accuracy_trAll)    

    # Merge all summaries and write them out to /tmp/tensorflow/mnist/logs
    # different writers are used to separate test accuracy from train accuracy
    # also a writer is implemented to observe CF after we trained on both sets
    merged = tf.summary.merge_all()

    #train_writer_ds = tf.summary.FileWriter(FLAGS.log_dir + '/training_ds',
    #                                        sess.graph)

    #test_writer_ds = tf.summary.FileWriter(FLAGS.log_dir + '/testing_ds')

    saver = tf.train.Saver(var_list=None)

    # Initialize all global variables or load model from pre-saved checkpoints
    # Open csv file for append when model is loaded, otherwise new file is created.
    if args.load_model:
        print('\nLoading Model: ', args.load_model)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=args.checkpoints_dir,
                                             latest_filename=args.load_model)

        saver.restore(sess=sess, save_path=args.checkpoints_dir + args.load_model + '.ckpt') ;
        # writer = csv.writer(open(FLAGS.plot_file, "a"))
    else:
        tf.global_variables_initializer().run()
    writer = csv.writer(open(FLAGS.plot_file, "wb"))

    with tf.name_scope("training"):
        print('\n\nTraining on given Dataset...')
        print('____________________________________________________________')
        print(time.strftime('%X %x %Z'))
        for i in range(FLAGS.start_at_step, FLAGS.max_steps + FLAGS.start_at_step):
            if i % LOG_FREQUENCY == 0:  # record summaries & test-set accuracy every 5 steps
                if testing_readout_layer is 1:
                    _lr, s, acc = sess.run([lr, merged, accuracy_tr1], feed_dict=feed_dict(False, i))
                elif testing_readout_layer is 2:
                    _lr, s, acc = sess.run([lr, merged, accuracy_tr2], feed_dict=feed_dict(False, i))
                elif testing_readout_layer is 3:
                    _lr, s, acc = sess.run([lr, merged, accuracy_tr3], feed_dict=feed_dict(False, i))
                elif testing_readout_layer is 4:
                    _lr, s, acc = sess.run([lr, merged, accuracy_tr4], feed_dict=feed_dict(False, i))
                elif testing_readout_layer is -1:
                    _lr, s, acc = sess.run([lr, merged, accuracy_trAll], feed_dict=feed_dict(False, i))            
                #test_writer_ds.add_summary(s, i)
                print(_lr, 'test set 1 accuracy at step: %s \t \t %s' % (i, acc))
                writer.writerow([i, acc])
            else:  # record train set summaries, and run training steps
                if training_readout_layer is 1:
                    s, _ = sess.run([merged, train_step_tr1], feed_dict(True, i))
                elif training_readout_layer is 2:
                    s, _ = sess.run([merged, train_step_tr2], feed_dict(True, i))
                if training_readout_layer is 3:
                    s, _ = sess.run([merged, train_step_tr3], feed_dict(True, i))
                if training_readout_layer is 4:
                    s, _ = sess.run([merged, train_step_tr4], feed_dict(True, i))
                #train_writer_ds.add_summary(s, i)
        #train_writer_ds.close()
        #test_writer_ds.close()

        if args.save_model:
            saver.save(sess=sess, save_path=args.checkpoints_dir + args.save_model + '.ckpt')


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir) and not FLAGS.load_model:
        tf.gfile.DeleteRecursively(FLAGS.log_dir + '/..')
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if FLAGS.train_classes:
        initDataSetsClasses()
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_classes', type=int, nargs='*',
                        help="Take only the specified Train classes from MNIST DataSet")
    parser.add_argument('--test_classes', type=int, nargs='*',
                        help="Take the specified Test classes from MNIST DataSet")

    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Number of steps to run trainer for given data set.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Keep probability for dropout.')

    parser.add_argument('--permuteTrain', type=int, default=-1,
                        help='Provide random seed for permutation train.')
    parser.add_argument('--permuteTest', type=int, default=-1,
                        help='Provide random seed for permutation test.')

    parser.add_argument('--hidden1', type=int, default=128,
                        help='Number of hidden units in layer 1')
    parser.add_argument('--hidden2', type=int, default=32,
                        help='Number of hidden units in layer 2')
    parser.add_argument('--hidden3', type=int, default=-1,
                        help='Number of hidden units in layer 3')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of mini-batches we feed from dataSet.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--decayStep', type=float, default=100000,
                        help='decayStep')
    parser.add_argument('--decayFactor', type=float, default=1.,
                        help='decayFactor')
    parser.add_argument('--load_model', type=str,
                        help='Load previously saved model. Leave empty if no model exists.')
    parser.add_argument('--save_model', type=str,
                        help='Provide path to save model.')
    parser.add_argument('--test_frequency', type=int, default='50',
                        help='Frequency after which a test cycle runs.')
    parser.add_argument('--start_at_step', type=int, default='0',
                        help='Global step should start here, and continue for the specified number of iterations')
    parser.add_argument('--training_readout_layer', type=int, default='1',
                        help='Specify the readout layer (1,2,3,4) for training.')
    parser.add_argument('--testing_readout_layer', type=int, default='1',
                        help='Specify the readout layer (1,2,3,4) for testing. Make sure this readout is already trained.')
    parser.add_argument('--plot_file', type=str,
                        default='convnet_more_layers.csv',
                        help='Filename for csv file to plot. Give .csv extension after file name.')
    parser.add_argument('--data_dir', type=str,
                        default='./',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default='./logs/',
                        help='Summaries log directory')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='./checkpoints/',
                        help='Checkpoints log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
