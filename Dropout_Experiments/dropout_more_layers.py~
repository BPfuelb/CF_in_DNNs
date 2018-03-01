from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utilities import * 

# implements LWTA catastrophic forgetting experiments of Srivastava et al. in Sec. 6


import tensorflow as tf
import numpy as np
import math

import argparse
import sys
import time
import csv
import os ;

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

def train(dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3):
    #args = parser.parse_args()
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
            k_h = FLAGS.dropout_hidden
            k_i = FLAGS.dropout_input
        else:
            xs, ys = dataSetTest.images, dataSetTest.labels
            k_h = 1.0
            k_i = 1.0
        if FLAGS.dnn_model=='fc':
          return {x: xs, y_: ys, global_step: i, keep_prob_input: k_i, keep_prob_hidden: k_h}
        elif FLAGS.dnn_model=='cnn':
          return {x: xs, y_: ys, global_step: i, keep_prob_input: k_i}
        else:
          return {x: xs, y_: ys, global_step: i}

    def feed_dict2(train, i):
        if train:
            xs, ys = dataSetTrain.next_batch(FLAGS.batch_size)
            k_h = FLAGS.dropout_hidden
            k_i = FLAGS.dropout_input
        else:
            xs, ys = dataSetTest2.images, dataSetTest2.labels
            k_h = 1.0
            k_i = 1.0
        if FLAGS.dnn_model=='fc':
          return {x: xs, y_: ys, global_step: i, keep_prob_input: k_i, keep_prob_hidden: k_h}
        elif FLAGS.dnn_model=='cnn':
          return {x: xs, y_: ys, global_step: i, keep_prob_input: k_i}
        else:
          return {x: xs, y_: ys, global_step: i}

    def feed_dict3(train, i):
        if train:
            xs, ys = dataSetTrain.next_batch(FLAGS.batch_size)
            k_h = FLAGS.dropout_hidden
            k_i = FLAGS.dropout_input
        else:
            xs, ys = dataSetTest3.images, dataSetTest3.labels
            k_h = 1.0
            k_i = 1.0
        if FLAGS.dnn_model=='fc':
          return {x: xs, y_: ys, global_step: i, keep_prob_input: k_i, keep_prob_hidden: k_h}
        elif FLAGS.dnn_model=='cnn':
          return {x: xs, y_: ys, global_step: i, keep_prob_input: k_i}
        else:
          return {x: xs, y_: ys, global_step: i}



    # weights initialization
    def weight_variable(shape, stddev, name="W"):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    # biases initialization
    def bias_variable(shape, name="b"):
        initial = tf.zeros(shape)
        return tf.Variable(initial, name=name)

    # define a fully connected layer
    def fc_layer(input, channels_in, channels_out, stddev, name='fc'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable([channels_in,
                                     channels_out], stddev)
            with tf.name_scope('biases'):
                b = bias_variable([channels_out])
            act = tf.nn.relu(tf.matmul(input, W) + b)
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", act)
            return act

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

    # weights initialization
    def weight_variable_cnn(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # biases initialization
    def bias_variable_cnn(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # define a 2d convolutional layer
    def conv_layer(input, channels_in, channels_out, name='conv'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable_cnn([5, 5, channels_in, channels_out])
            with tf.name_scope('biases'):
                b = bias_variable([channels_out])
            conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
            act = tf.nn.relu(conv + b)
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", act)
            return act

    # define a fully connected layer
    def fc_layer_cnn(input, channels_in, channels_out, name='fc'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable_cnn([channels_in, channels_out])
            with tf.name_scope('biases'):
                b = bias_variable([channels_out])
            act = tf.nn.relu(tf.matmul(input, W) + b)
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", act)
            return act

    # define a readout layer
    def ro_layer_cnn(input, channels_in, channels_out, name='read'):
        with tf.name_scope(name):
            with tf.name_scope('weights'):
                W = weight_variable_cnn([channels_in, channels_out])
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
    config.gpu_options.allow_growth=True
    config.log_device_placement=False ;
    # sess = tf.Session(config=config)
    sess = tf.InteractiveSession(config=config)

    # Placeholder for input variables
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

    global global_step
    global_step = tf.placeholder(tf.float32, shape=[], name="step")
    logits_tr1 = None; logits_tr2 = None ; logits_tr4 = None ; logitsAll = None ;
    keep_prob_input = tf.placeholder(tf.float32)

    if FLAGS.dnn_model=="fc":
      # apply dropout to the input layer
      #tf.summary.scalar('dropout_input', keep_prob_input)

      x_drop = tf.nn.dropout(x, keep_prob_input)

      # Create the first hidden layer
      h_fc1 = fc_layer(x_drop, IMAGE_PIXELS, FLAGS.hidden1,
                     1.0 / math.sqrt(float(IMAGE_PIXELS)), 'h_fc1')

      # Apply dropout to first hidden layer
      keep_prob_hidden = tf.placeholder(tf.float32)
      tf.summary.scalar('dropout_hidden', keep_prob_hidden)

      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_hidden)

      # Create the second hidden layer
      h_fc2 = fc_layer(h_fc1_drop, FLAGS.hidden1, FLAGS.hidden2,
                     1.0 / math.sqrt(float(FLAGS.hidden1)), 'h_fc2')

      # Apply dropout to second hidden layer
      h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_hidden)

      # Create a softmax linear classification layer for the outputs
      if FLAGS.hidden3 == -1:
          logits_tr1 = softmax_linear(h_fc2_drop, FLAGS.hidden2, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2)),
                                    'softmax_linear_tr1')
          logits_tr2 = softmax_linear(h_fc2_drop, FLAGS.hidden2, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2)),
                                    'softmax_linear_tr2')
          logits_tr3 = softmax_linear(h_fc2_drop, FLAGS.hidden2, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2)),
                                    'softmax_linear_tr3')
          logits_tr4 = softmax_linear(h_fc2_drop, FLAGS.hidden2, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden2)),
                                    'softmax_linear_tr4')
      else:
          h_fc3 = fc_layer(h_fc2_drop, FLAGS.hidden2, FLAGS.hidden3,
                         1.0 / math.sqrt(float(FLAGS.hidden3)), 'h_fc3')

          # Apply dropout to third hidden layer
          h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob_hidden)

          logits_tr1 = softmax_linear(h_fc3_drop, FLAGS.hidden3, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3)),
                                    'softmax_linear_tr1')
          logits_tr2 = softmax_linear(h_fc3_drop, FLAGS.hidden3, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3)),
                                    'softmax_linear_tr2')
          logits_tr3 = softmax_linear(h_fc3_drop, FLAGS.hidden3, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3)),
                                    'softmax_linear_tr3')
          logits_tr4 = softmax_linear(h_fc3_drop, FLAGS.hidden3, NUM_CLASSES,
                                    1.0 / math.sqrt(float(FLAGS.hidden3)),
                                    'softmax_linear_tr4')

      logitsAll = logits_tr1 + logits_tr2 + logits_tr3 + logits_tr4;
    elif FLAGS.dnn_model=="cnn":
      print ("-------------------------------------------------------------------------------------------------**********");
      x_image = tf.reshape(x, [-1, 28, 28, 1])
      #tf.summary.image('input', x_image, 9)

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
      h_fc1 = fc_layer_cnn(h_pool2_flattened, 7 * 7 * 64, 1024, 'h_fc1')

      # Apply dropout to the densely connected layer      
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_input)

      # Create a softmax linear classification layer for the outputs
      logits_tr1 = ro_layer_cnn(h_fc1_drop, 1024, 10, 'ro_layer_tr1')
      logits_tr2 = ro_layer_cnn(h_fc1_drop, 1024, 10, 'ro_layer_tr2')
      logits_tr3 = ro_layer_cnn(h_fc1_drop, 1024, 10, 'ro_layer_tr3')
      logits_tr4 = ro_layer_cnn(h_fc1_drop, 1024, 10, 'ro_layer_tr4')
      logitsAll = logits_tr1 + logits_tr2 + logits_tr3 + logits_tr4 ;
      
    elif FLAGS.dnn_model=="lwta":    
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

      logitsAll = logits_tr1 + logits_tr2 + logits_tr3 + logits_tr4;
    else:
      print ("invalid model")
      sys.exit(-1) ;


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
            correct_prediction_trAll = tf.equal(tf.argmax(logitsAll, 1), tf.argmax(y_, 1))
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
    writer2 = None; writer3 = None ;
    if FLAGS.test2_classes != None:
      print ("WRITER2=",FLAGS.plot_file2) ;
      writer2 = csv.writer(open(FLAGS.plot_file2, "wb"))
    if FLAGS.test3_classes != None:
      print ("WRITER3=",FLAGS.plot_file3) ;
      writer3 = csv.writer(open(FLAGS.plot_file3, "wb"))

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
                    _lr, s, acc, l1, l2, l3, l4, lAll = sess.run(
                        [lr, merged, accuracy_trAll, logits_tr1, logits_tr2, logits_tr3, logits_tr4, logitsAll],
                        feed_dict=feed_dict(False, i))
                #test_writer_ds.add_summary(s, i)
                print(_lr, 'test set 1 accuracy at step: %s \t \t %s' % (i, acc))
                writer.writerow([i, acc])

                if FLAGS.test2_classes != None:
                  if FLAGS.testing2_readout_layer is 1:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr1], feed_dict=feed_dict2(False, i))
                  elif FLAGS.testing2_readout_layer is 2:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr2], feed_dict=feed_dict2(False, i))
                  elif FLAGS.testing2_readout_layer is 3:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr3], feed_dict=feed_dict2(False, i))
                  elif FLAGS.testing2_readout_layer is 4:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr4], feed_dict=feed_dict2(False, i))
                  elif FLAGS.testing2_readout_layer is -1:
                      _lr, s, acc, l1, l2, l3, l4, lAll = sess.run(
                        [lr, merged, accuracy_trAll, logits_tr1, logits_tr2, logits_tr3, logits_tr4, logitsAll],
                        feed_dict=feed_dict2(False, i))
                  print(_lr, 'test set 2 accuracy at step: %s \t \t %s' % (i, acc))
                  writer2.writerow([i, acc])

                if FLAGS.test3_classes != None:
                  if FLAGS.testing3_readout_layer is 1:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr1], feed_dict=feed_dict3(False, i))
                  elif FLAGS.testing3_readout_layer is 2:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr2], feed_dict=feed_dict3(False, i))
                  elif FLAGS.testing3_readout_layer is 3:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr3], feed_dict=feed_dict3(False, i))
                  elif FLAGS.testing3_readout_layer is 4:
                      _lr, s, acc = sess.run([lr, merged, accuracy_tr4], feed_dict=feed_dict3(False, i))
                  elif FLAGS.testing3_readout_layer is -1:
                      _lr, s, acc, l1, l2, l3, l4, lAll = sess.run(
                        [lr, merged, accuracy_trAll, logits_tr1, logits_tr2, logits_tr3, logits_tr4, logitsAll],
                        feed_dict=feed_dict3(False, i))
                  print(_lr, 'test set 3 accuracy at step: %s \t \t %s' % (i, acc))
                  writer3.writerow([i, acc])



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
            print ("saving to", args.checkpoints_dir + args.save_model + '.ckpt')
            saver.save(sess=sess, save_path=args.checkpoints_dir + args.save_model + '.ckpt')


def main(_):
    if FLAGS.permuteTrain is 0 or FLAGS.permuteTrain:
        print("Permutation!!!!!!!!!!!!!")
    if tf.gfile.Exists(FLAGS.log_dir) and not FLAGS.load_model:
        #tf.gfile.DeleteRecursively(FLAGS.log_dir + '/..')
        #tf.gfile.MakeDirs(FLAGS.log_dir)
        pass ;
    if FLAGS.train_classes:
      dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3 = initDataSetsClasses(FLAGS) ;
      train(dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3)


if __name__ == '__main__':
    parser = createParser ();
    FLAGS, unparsed = parser.parse_known_args()
    print ("TEST2=",FLAGS.test2_classes) ;
    print ("--",FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
