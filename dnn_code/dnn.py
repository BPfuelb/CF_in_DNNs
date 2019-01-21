"""
Code for FC, D-FC, CONV, D-CONV and LWTA models with parameters (FLAGS)

  structure of file:
    0. load data from disk
    1. definition of functions to build DNN (model) (distinguished by model)
    2. build DNN
    (2,5.) if necessary load stored model for further training
    3. train and evaluate (on multiple test datasets, save results to csv-files) 
    (3,5.) store model
    
@author: BPF (modified)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from functools import reduce
import math
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from dataset.dataset import load_dataset
import defaultParser
import tensorflow as tf

FLAGS = None

def train(dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3, properties):
  ''' 
    * train and evaluate only one dataset (D1)
    * train one dataset and evaluate 3 datasets e.g. train (D2) evaluate (D2, D1, D1uD2)
  
  @param dataSetTrain: first dataset to train (tf.Dataset) e.g. D2 
  @param dataSetTest: first dataset to test (tf.Dataset) e.g. D2
  @param dataSetTest2: second dataset to test (tf.Dataset) e.g. D1
  @param dataSetTest3: third dataset to test (tf.Dataset) eg.g D1uD2
  @param properties: properties for first dataset (dict)
  '''
  
  # FLAGS
  BATCH_SIZE = FLAGS.batch_size
  EPOCHS = FLAGS.epochs
  MEASURING_POINTS = FLAGS.measuring_points
  CHECKPOINTS_DIR = FLAGS.checkpoints_dir
  DROPOUT_HIDDEN = FLAGS.dropout_hidden
  DROPOUT_INPUT = FLAGS.dropout_input
  DNN_MODEL = FLAGS.dnn_model
  LOAD_MODEL = FLAGS.load_model
  SAVE_MODEL = FLAGS.save_model
  HIDDEN1 = FLAGS.hidden1
  HIDDEN2 = FLAGS.hidden2
  HIDDEN3 = FLAGS.hidden3
  LEARNING_RATE = FLAGS.learning_rate
  
  # LWTA
  LWTA_BLOCK_SIZE = FLAGS.lwtaBlockSize
  
  PLOT_FILE1 = FLAGS.plot_file
  PLOT_FILE2 = FLAGS.plot_file2
  PLOT_FILE3 = FLAGS.plot_file3
  
  NUM_CLASSES = properties['num_classes']
  NUM_OF_CHANNELS = properties['num_of_channels']
  IMAGE_SIZE = properties['dimensions']
  IMAGE_PIXELS = reduce(lambda x, y: x * y, IMAGE_SIZE) * NUM_OF_CHANNELS
  
  if not os.path.exists(CHECKPOINTS_DIR): os.makedirs(CHECKPOINTS_DIR)

  def feed_dict(test_set=None):
    ''' feed dictionary for all datasets '''
    if not test_set:
      xs, ys = dataSetTrain.next_batch(BATCH_SIZE)
      k_h = DROPOUT_HIDDEN
      k_i = DROPOUT_INPUT
    else:
      if test_set == 'test1': xs, ys = dataSetTest.next_batch(BATCH_SIZE)
      if test_set == 'test2': xs, ys = dataSetTest2.next_batch(BATCH_SIZE)
      if test_set == 'test3': xs, ys = dataSetTest3.next_batch(BATCH_SIZE)
      k_h = 1.0
      k_i = 1.0
    f_dict = {x: xs, y_: ys}
    if DNN_MODEL == 'fc' : f_dict.update({keep_prob_input: k_i, keep_prob_hidden: k_h}) 
    if DNN_MODEL == 'cnn': f_dict.update({keep_prob_input: k_i})
    return f_dict

  def weight_variable(shape, stddev, name="W"):
    """ weights initialization """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

  def bias_variable(shape, name="b"):
      """ biases initialization """
      initial = tf.zeros(shape)
      return tf.Variable(initial, name=name)

  def fc_layer(x, channels_in, channels_out, stddev, name='fc'):
    """ define a fully connected layer """
    with tf.name_scope(name):
      with tf.name_scope('weights'):
        W = weight_variable([channels_in, channels_out], stddev)
      with tf.name_scope('biases'):
        b = bias_variable([channels_out])
      act = tf.nn.relu(tf.matmul(x, W) + b)
      tf.summary.histogram("weights", W)
      tf.summary.histogram("biases", b)
      tf.summary.histogram("activation", act)
      return act

  def lwta_layer(x, channels_in, channels_out, blockSize, stddev, name='lwta'):
    """ careful: actual output size is channels_out*blockSize """
    with tf.name_scope(name):
      with tf.name_scope('weights'):
        W = weight_variable([channels_in, channels_out * blockSize], stddev)
      with tf.name_scope('biases'):
        b = bias_variable([blockSize * channels_out])
      actTmp = tf.reshape(tf.matmul(x, W) + b, (-1, channels_out, blockSize))
      # ask where the maxes lie --> bool tensor, then cast to float where True --> 1.0, False --> 0.0
      maxesMask = tf.cast(tf.equal(actTmp, tf.expand_dims(tf.reduce_max(actTmp, axis=2), 2)), tf.float32)
      # point-wise multiplication of these two tensors will result in a tesnor where only the max
      # activations are left standing, the rest goes to 0 as specified by LWTA
      return tf.reshape(actTmp * maxesMask, (-1, channels_out * blockSize))

  def softmax_linear(x, channels_in, channels_out, stddev, name='read'):
    """ define a sotfmax linear classification layer """
    with tf.name_scope(name):
      with tf.name_scope('weights'):
        W = weight_variable([channels_in, channels_out], stddev, name="WMATRIX")
      with tf.name_scope('biases'):
        b = bias_variable([channels_out], name="bias")
      act = tf.matmul(x, W) + b
      tf.summary.histogram("weights", W)
      tf.summary.histogram("biases", b)
      tf.summary.histogram("activation", act)
      return act

  def weight_variable_cnn(shape):
    """ weights initialization """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable_cnn(shape):
    """ biases initialization """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv_layer(x, channels_in, channels_out, name='conv'):
    """ define a 2d convolutional layer """
    with tf.name_scope(name):
      with tf.name_scope('weights'):
        W = weight_variable_cnn([5, 5, channels_in, channels_out])
      with tf.name_scope('biases'):
        b = bias_variable([channels_out])
      conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
      act = tf.nn.relu(conv + b)
      tf.summary.histogram("weights", W)
      tf.summary.histogram("biases", b)
      tf.summary.histogram("activation", act)
      return act

  def fc_layer_cnn(x, channels_in, channels_out, name='fc'):
    """ define a fully connected layer """
    with tf.name_scope(name):
      with tf.name_scope('weights'):
        W = weight_variable_cnn([channels_in, channels_out])
      with tf.name_scope('biases'):
        b = bias_variable([channels_out])
      act = tf.nn.relu(tf.matmul(x, W) + b)
      tf.summary.histogram("weights", W)
      tf.summary.histogram("biases", b)
      tf.summary.histogram("activation", act)
      return act

  def ro_layer_cnn(x, channels_in, channels_out, name='read'):
    """ define a readout layer """
    with tf.name_scope(name):
      with tf.name_scope('weights'):
        W = weight_variable_cnn([channels_in, channels_out])
      with tf.name_scope('biases'):
        b = bias_variable([channels_out])
      act = tf.matmul(x, W) + b
      tf.summary.histogram("weights", W)
      tf.summary.histogram("biases", b)
      tf.summary.histogram("activation", act)
      return act

  def max_pool_2x2(x):
    """ pooling """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Start an Interactive session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.log_device_placement = False
  sess = tf.InteractiveSession(config=config)

  # Placeholder for input variables
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='labels')

  keep_prob_input = tf.placeholder(tf.float32)
  
  logits_tr = None

  if DNN_MODEL == "fc":
    keep_prob_hidden = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_hidden', keep_prob_hidden)
    
    x_drop = tf.nn.dropout(x, keep_prob_input)

    # Create the first hidden layer and apply dropout
    h_fc1 = fc_layer(x_drop, IMAGE_PIXELS, HIDDEN1, 1.0 / math.sqrt(float(IMAGE_PIXELS)), 'h_fc1')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_hidden)

    # Create the second hidden layer and apply dropout
    h_fc2 = fc_layer(h_fc1_drop, HIDDEN1, HIDDEN2, 1.0 / math.sqrt(float(HIDDEN1)), 'h_fc2')
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_hidden)

    # Create a softmax linear classification layer for the outputs
    if HIDDEN3 == -1:
      logits_tr = softmax_linear(h_fc2_drop, HIDDEN2, NUM_CLASSES, 1.0 / math.sqrt(float(HIDDEN2)), 'softmax_linear_tr')
    else:
      # Create the third hidden layer and apply dropout
      h_fc3 = fc_layer(h_fc2_drop, HIDDEN2, HIDDEN3, 1.0 / math.sqrt(float(HIDDEN3)), 'h_fc3')
      h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob_hidden)

      logits_tr = softmax_linear(h_fc3_drop, HIDDEN3, NUM_CLASSES, 1.0 / math.sqrt(float(HIDDEN3)), 'softmax_linear_tr')

  elif DNN_MODEL == "cnn":

    def calcDims(oldx, oldy, fx, fy, stepx, stepy):
      newx = ((oldx - 0 * fx) / stepx) + 0
      newy = ((oldy - 0 * fy) / stepy) + 0

      newx = (newx + 1) // 2
      newy = (newy + 1) // 2
      return newx, newy

    x_image = tf.reshape(x, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_OF_CHANNELS])

    # Create the first convolutional layer
    # convolve x_image with the weight tensor, add the bias, apply ReLu
    h_conv1 = conv_layer(x_image, NUM_OF_CHANNELS, 32, 'h_conv1')
    # max pooling for first conv layer
    h_pool1 = max_pool_2x2(h_conv1)
    newx, newy = calcDims(IMAGE_SIZE[0], IMAGE_SIZE[1], 5, 5, 1, 1) 

    # Create the second convolutional layer
    h_conv2 = conv_layer(h_pool1, 32, 64, 'h_conv2')
    # max pooling for second conv layer
    h_pool2 = max_pool_2x2(h_conv2)
    newx, newy = calcDims(newx, newy, 5, 5, 1, 1) 

    # reshape tensor from the pooling layer into a batch of vectors
    h_pool2_flattened = tf.reshape(h_pool2, [-1, int(newx) * int(newy) * 64])

    # Create a densely Connected Layer
    # image size reduced to 7x7, add a fully-connected layer with 1024 neurons
    # to allow processing on the entire image.
    h_fc1 = fc_layer_cnn(h_pool2_flattened, int(newx) * int(newy) * 64, 1024, 'h_fc1')

    # Apply dropout to the densely connected layer
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_input)

    # Create a softmax linear classification layer for the outputs
    logits_tr = ro_layer_cnn(h_fc1_drop, 1024, NUM_CLASSES, 'ro_layer_tr')

  elif DNN_MODEL == "lwta":
    ''' implements LWTA catastrophic forgetting experiments of Srivastava et al. in Sec. 6 '''
    # Create the first hidden layer
    h_fc1 = lwta_layer(x, IMAGE_PIXELS, HIDDEN1, LWTA_BLOCK_SIZE, 1.0 / math.sqrt(float(IMAGE_PIXELS)), 'h_lwta1')

    # Create the second hidden layer
    h_fc2 = lwta_layer(h_fc1, HIDDEN1 * LWTA_BLOCK_SIZE, HIDDEN2, LWTA_BLOCK_SIZE, 1.0 / math.sqrt(float(HIDDEN1 * LWTA_BLOCK_SIZE)), 'h_lwta2')

    # Create a softmax linear classification layer for the outputs
    if HIDDEN3 == -1:
      logits_tr = softmax_linear(h_fc2, HIDDEN2 * LWTA_BLOCK_SIZE, NUM_CLASSES, 1.0 / math.sqrt(float(HIDDEN2 * LWTA_BLOCK_SIZE)), 'softmax_linear_tr')
    else:
      h_fc3 = lwta_layer(h_fc2, HIDDEN2 * LWTA_BLOCK_SIZE, HIDDEN3, LWTA_BLOCK_SIZE, 1.0 / math.sqrt(float(HIDDEN3 * LWTA_BLOCK_SIZE)), 'h_lwta3')
      logits_tr = softmax_linear(h_fc3, HIDDEN3 * LWTA_BLOCK_SIZE, NUM_CLASSES, 1.0 / math.sqrt(float(HIDDEN3 * LWTA_BLOCK_SIZE)), 'softmax_linear_tr')

  else:
    print ("invalid model")
    sys.exit(-1)

  # Define the loss model as a cross entropy with softmax layer 1
  with tf.name_scope('cross_entropy_tr'):
    diff_tr = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits_tr)
    with tf.name_scope('total_tr'):
      cross_entropy_tr = tf.reduce_mean(diff_tr)

  # Use MomentumOptimizer optimizer for training steps and minimize cross-entropy
  with tf.name_scope('train_tr'):
    train_step_tr = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.99).minimize(cross_entropy_tr)

  # Compute correct prediction and accuracy
  with tf.name_scope('accuracy_tr'):
    with tf.name_scope('correct_prediction_tr'):
      correct_prediction_tr = tf.equal(tf.argmax(logits_tr, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy_tr'):
      accuracy_tr = tf.reduce_mean(tf.cast(correct_prediction_tr, tf.float32))
  tf.summary.scalar('accuracy_tr', accuracy_tr)

  saver = tf.train.Saver(var_list=None)

  # Initialize all global variables or load model from pre-saved checkpoints
  # Open csv file for append when model is loaded, otherwise new file is created.
  if LOAD_MODEL:
    print('\nLoading Model: ', LOAD_MODEL)
    tf.train.get_checkpoint_state(checkpoint_dir=CHECKPOINTS_DIR, latest_filename=LOAD_MODEL)

    saver.restore(sess=sess, save_path='{}{}.ckpt'.format(CHECKPOINTS_DIR, LOAD_MODEL))
  else:
    tf.global_variables_initializer().run()
    
  csv_writer1 = csv.writer(open(PLOT_FILE1, 'w'))
  print ('WRITER1= {}'.format(PLOT_FILE1))
  if PLOT_FILE2 != None:
      print ('WRITER2= {}'.format(PLOT_FILE2))
      csv_writer2 = csv.writer(open(PLOT_FILE2, 'w'))
  if PLOT_FILE3 != None:
      print ('WRITER3= {}'.format(PLOT_FILE3))
      csv_writer3 = csv.writer(open(PLOT_FILE3, 'w'))
  
  with tf.name_scope("training"):
    print('\n\nTraining on given Dataset...')
    print('____________________________________________________________')

    num_batches = math.ceil(dataSetTrain.images.shape[0] / BATCH_SIZE)              # number of batches for one epoch (round up)
    num_batches_test1_epoch = math.ceil(dataSetTest.images.shape[0] / BATCH_SIZE)   # number of batches for one epoch (round up)
    num_batches_test2_epoch = math.ceil(dataSetTest2.images.shape[0] / BATCH_SIZE)  # number of batches for one epoch (round up)
    num_batches_test3_epoch = math.ceil(dataSetTest3.images.shape[0] / BATCH_SIZE)  # number of batches for one epoch (round up)
    num_batches_all_epochs = int(num_batches * EPOCHS)                              # number of all batches for all epochs
    test_all_n_batches = num_batches_all_epochs // MEASURING_POINTS                 # test all n iterations (round)
    if test_all_n_batches == 0: test_all_n_batches = 1                              # if less than MEASURING_POINTS iterations for all epochs are done
     
    for epoch in range(EPOCHS):
      print('epoch number: {}'.format(epoch + 1))
      for batch in range(num_batches):
        # training
        sess.run([train_step_tr], feed_dict())
        
        # test: TODO: test function?
        if batch % test_all_n_batches == 0:
          acc = 0.0
          for _ in range(num_batches_test1_epoch): # test one epoch for test set 1
            acc += sess.run(accuracy_tr, feed_dict=feed_dict('test1'))
          res = [batch, acc / num_batches_test1_epoch]
          print('test set 1 accuracy at step: {} \t \t {}'.format(*res))
          csv_writer1.writerow(res)
          
          if PLOT_FILE2 != None:
            acc = 0.0
            for _ in range(num_batches_test2_epoch): # test one epoch for test set 2
              acc += sess.run(accuracy_tr, feed_dict=feed_dict('test2'))
            
            res = [batch, acc / num_batches_test2_epoch]
            print('test set 2 accuracy at step: {} \t \t {}'.format(*res))
            csv_writer2.writerow(res)

          if PLOT_FILE3 != None:
            acc = 0.0
            for _ in range(num_batches_test3_epoch): # test one epoch for test set 3
              acc += sess.run(accuracy_tr, feed_dict=feed_dict('test3'))
            res = [batch, acc / num_batches_test3_epoch]
            print('test set 3 accuracy at step: {} \t \t {}'.format(*res))
            csv_writer3.writerow(res)
      # for batch in range(num_batches):
    # for epoch in range(EPOCHS):
      
    if SAVE_MODEL:
      print('saving to {}{}.ckpt'.format(CHECKPOINTS_DIR, SAVE_MODEL))
      saver.save(sess=sess, save_path='{}/{}.ckpt'.format(CHECKPOINTS_DIR, SAVE_MODEL))


def main(_):
    defaultParser.printFlags(FLAGS)

    if FLAGS.permuteTrain is 0 or FLAGS.permuteTrain: print("Permutation!!!!!!!!!!!!!")
    
    if FLAGS.train_classes:
        dataSetTrain, _, dataSetTest, dataSetTest2, dataSetTest3, properties = load_dataset(FLAGS)
        train(dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3, properties)


if __name__ == '__main__':
  parser = defaultParser.create_default_parser()

  #----------------------------------------------------------- LWTA PARAMETERS
  parser.add_argument('--lwtaBlockSize', type=int,
                      default=2,
                      help='Number of lwta blocks in all hidden layers')

  #----------------------------------------------------------------- DNN MODEL
  parser.add_argument('--dnn_model', type=str,
                      default='fc',
                      help='''which dn type is used?
                      fc
                      cnn
                      lwta
                      ''')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
