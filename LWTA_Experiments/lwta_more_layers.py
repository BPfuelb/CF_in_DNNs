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
    args = FLAGS ;
    training_readout_layer = args.training_readout_layer
    testing_readout_layer = args.testing_readout_layer
    LOG_FREQUENCY = args.test_frequency
    if not os.path.exists("./checkpoints"):
      os.mkdir("./checkpoints")


    # feed dictionary for dataSetOne
    def feed_dict(train, i):
        if train:
            xs, ys = dataSetTrain.next_batch(FLAGS.batch_size)
        else:
            xs, ys = dataSetTest.images, dataSetTest.labels
        return {x: xs, y_: ys, global_step: i}

    # weights initialization
    def weight_variable(shape, stddev, name="W"):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    # biases initialization
    def bias_variable(shape, name="b"):
        initial = tf.zeros(shape)
        return tf.Variable(initial, name=name)

    # careful: actual output size is channels_out*blockSize
    def lwta_layer(input, channels_in, channels_out, blockSize, stddev, name='lwta'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable([channels_in,
                                     channels_out * blockSize], stddev)
            with tf.name_scope('biases'):
                b = bias_variable([blockSize * channels_out])
            actTmp = tf.reshape(tf.matmul(input, W) + b, (-1, channels_out, blockSize));
            # ask where the maxes lie --> bool tensor, then cast to float where True --> 1.0, False --> 0.0
            maxesMask = tf.cast(tf.equal(actTmp, tf.expand_dims(tf.reduce_max(actTmp, axis=2), 2)), tf.float32)
            # point-wise multiplication of these two tensors will result in a tesnor where only the max
            # activations are left standing, the rest goes to 0 as specified by LWTA
            return tf.reshape(actTmp * maxesMask, (-1, channels_out * blockSize))

    # define a sotfmax linear classification layer
    def softmax_linear(input, channels_in, channels_out, stddev, name='read'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable([channels_in,
                                     channels_out], stddev, name="WMATRIX")
                wdict[W] = W;
            with tf.name_scope('biases'):
                b = bias_variable([channels_out], name="bias")
            act = tf.matmul(input, W) + b
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", act)
            return act

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

    global global_step
    global_step = tf.placeholder(tf.float32, shape=[], name="step");

    # Create the first hidden layer
    h_fc1 = lwta_layer(x, IMAGE_PIXELS, FLAGS.hidden1, FLAGS.lwtaBlockSize,
                       1.0 / math.sqrt(float(IMAGE_PIXELS)), 'h_lwta1')

    # Create the second hidden layer
    h_fc2 = lwta_layer(h_fc1, FLAGS.hidden1 * FLAGS.lwtaBlockSize, FLAGS.hidden2, FLAGS.lwtaBlockSize,
                       1.0 / math.sqrt(float(FLAGS.hidden1 * FLAGS.lwtaBlockSize)), 'h_lwta2')

    # Create a softmax linear classification layer for the outputs
    if FLAGS.hidden3 == -1:
        logits_tr1 = softmax_linear(h_fc2, FLAGS.hidden2 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr1')
        logits_tr2 = softmax_linear(h_fc2, FLAGS.hidden2 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr2')
        logits_tr3 = softmax_linear(h_fc2, FLAGS.hidden2 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr3')
        logits_tr4 = softmax_linear(h_fc2, FLAGS.hidden2 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr4')
    else:
        h_fc3 = lwta_layer(h_fc2, FLAGS.hidden2 * FLAGS.lwtaBlockSize, FLAGS.hidden3, FLAGS.lwtaBlockSize,
                           1.0 / math.sqrt(float(FLAGS.hidden3 * FLAGS.lwtaBlockSize)), 'h_lwta3')

        logits_tr1 = softmax_linear(h_fc3, FLAGS.hidden3 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr1')
        logits_tr2 = softmax_linear(h_fc3, FLAGS.hidden3 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr2')
        logits_tr3 = softmax_linear(h_fc3, FLAGS.hidden3 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr3')
        logits_tr4 = softmax_linear(h_fc3, FLAGS.hidden3 * FLAGS.lwtaBlockSize, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3 * FLAGS.lwtaBlockSize)),
                                    'softmax_linear_tr4')

    logits_trAll = logits_tr1 + logits_tr2 + logits_tr3 + logits_tr4;

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

        train_step_tr1 = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.99).minimize(cross_entropy_tr1)

    with tf.name_scope('train_tr2'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        FLAGS.decayStep, FLAGS.decayFactor, staircase=True)

        train_step_tr2 = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.99).minimize(cross_entropy_tr2)

    with tf.name_scope('train_tr3'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        FLAGS.decayStep, FLAGS.decayFactor, staircase=True)

        train_step_tr3 = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.99).minimize(cross_entropy_tr3)

    with tf.name_scope('train_tr4'):
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        FLAGS.decayStep, FLAGS.decayFactor, staircase=True)

        train_step_tr4 = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.99).minimize(cross_entropy_tr4)

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
        #tf.gfile.DeleteRecursively(FLAGS.log_dir + '/..')
        #tf.gfile.MakeDirs(FLAGS.log_dir)
        pass ;
    if FLAGS.train_classes:
        dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3 = initDataSetsClasses()
    train(dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3)


if __name__ == '__main__':
    parser = createParser() ;
    parser.add_argument('--lwtaBlockSize', type=int, default=2,
                        help='Number of lwta blocks in all hidden layers')

    """
    parser.add_argument('--train_classes', type=int, nargs='*',
                        help="Take only the specified Train classes from MNIST DataSet")
    parser.add_argument('--test_classes', type=int, nargs='*',
                        help="Take the specified Test classes from MNIST DataSet")

    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Number of steps to run trainer for given data set.')

    parser.add_argument('--permuteTrain', type=int, default=-1,
                        help='Provide random seed for permutation train.')
    parser.add_argument('--permuteTest', type=int, default=-1,
                        help='Provide random seed for permutation test.')

    parser.add_argument('--dropout_hidden', type=float, default=0.5,
                        help='Keep probability for dropout on hidden units.')
    parser.add_argument('--dropout_input', type=float, default=0.8,
                        help='Keep probability for dropout on input units.')

    parser.add_argument('--hidden1', type=int, default=128,
                        help='Number of hidden units in layer 1')
    parser.add_argument('--hidden2', type=int, default=32,
                        help='Number of hidden units in layer 2')
    parser.add_argument('--hidden3', type=int, default=-1,
                        help='Number of hidden units in layer 3')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of mini-batches we feed from dataSetOne.')
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
                        default='lwta_more_layers.csv',
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
    """

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
