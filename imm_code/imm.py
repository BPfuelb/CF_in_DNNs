"""
Implements L2-IMM and wtIMM strategies, using either mean and/or mode-imm
This is the only imm script that should be used, no longer do we divide functions
into main, main_l2 and main_l2_drop scripts. Distinction ebtween l2 and wt transfer is
done by looking at the value of the regularizer parameter, if it is 0.0 --> wt transfer else l2_transfer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import model_utils
from model import imm
from functools import reduce
import os
import sys
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from dataset.dataset import load_dataset
import defaultParser
import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
  ''' ... '''
  # FLAGS
  mean_imm = FLAGS.mean_imm
  mode_imm = FLAGS.mode_imm
  lmbda = FLAGS.regularizer
  optimizer = FLAGS.optimizer
  learning_rate = FLAGS.learning_rate
  alphaStartStopStep = FLAGS.alphaStartStopStep
  no_of_task = FLAGS.tasks
  no_of_node = [] # TODO: WTF?
  keep_prob_info = []

  dataSetTrain, dataSetTrain2, dataSetTest, dataSetTest2, dataSetTest3 , properties = load_dataset(FLAGS)

  NUM_CLASSES = properties['num_classes']
  NUM_OF_CHANNELS = properties['num_of_channels']
  IMAGE_SIZE = properties['dimensions']
  IMAGE_PIXELS = reduce(lambda x, y: x * y, IMAGE_SIZE) * NUM_OF_CHANNELS

  if FLAGS.hidden3 == -1:
    no_of_node = [IMAGE_PIXELS, FLAGS.hidden1, FLAGS.hidden2, NUM_CLASSES]
    keep_prob_info = [0.8, 0.5, 0.5]
  else:
    no_of_node = [IMAGE_PIXELS, FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3, NUM_CLASSES]
    keep_prob_info = [0.8, 0.5, 0.5, 0.5]
    keep_prob_info = [1, 1, 1, 1] # TODO: WTF?

  plotfile = None
  plotfile2 = None
  plotfile3 = None

  if (FLAGS.plot_file is not None):  plotfile = open(FLAGS.plot_file, "w")
  if (FLAGS.plot_file2 is not None): plotfile2 = open(FLAGS.plot_file2, "w")
  if (FLAGS.plot_file3 is not None): plotfile3 = open(FLAGS.plot_file3, "w")

  # convention: appended _ means test data, x is data, y is labels
  datasets = {'train1':dataSetTrain,
              'train2': dataSetTrain2,
              #---------------------
              'test1': dataSetTest,
              'test2': dataSetTest2,
              'test3': dataSetTest3 }
  start_time = time.time()

  with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    mlp = imm.TransferNN(no_of_node,
                         (optimizer, learning_rate),
                         keep_prob_info=keep_prob_info,
                         datasets=datasets,
                         flags=FLAGS)
    # l2-specific
    if FLAGS.regularizer > 0: mlp.RegPatch(lmbda)
    
    sess.run(tf.global_variables_initializer())

    if FLAGS.tasks == 0:
      print ('Baseline')
      # train and test on baseline
      mlp.Train(sess, 'train1', 'test1', logTo=plotfile)
      print('Used Time: {:.4f}s'.format(time.time() - start_time))
      sys.exit(0)

    # copies of all layers for each task
    L_copy = []

    # fisher matrices for each task
    FM = []

    print('================= Train #1 ({}) ================'.format(optimizer))

    # training
    mlp.Train(sess, 'train1', 'test1', logTo=plotfile)
    
    if FLAGS.tasks < 2:
      sys.exit()
    
    if mean_imm or mode_imm:
      print('copy layer values (1)')
      L_copy.append(model_utils.CopyLayerValues(sess, mlp.Layers))

    if mode_imm:
      print('calculate fisher matrix (1)')
      FM.append(mlp.CalculateFisherMatrix(sess, datasets['train1']))

    print('================= Train #2 ({}) ================'.format(optimizer))
    # l2 specific # regularization from weight of pre-task
    if FLAGS.regularizer > 0: model_utils.CopyLayers(sess, mlp.Layers, mlp.Layers_reg)

    # training
    mlp.Train(sess, 'train2', 'test2', logTo=plotfile2)

    if mean_imm or mode_imm:
      print('copy layer values (2)')
      L_copy.append(model_utils.CopyLayerValues(sess, mlp.Layers))

    if mode_imm:
      print('calculate fisher matrix (2)')
      FM.append(mlp.CalculateFisherMatrix(sess, datasets['train2']))

    # end training
    if no_of_task < 1: return print('QUITTING')

    # now, after learning 2 tasks, we merge the two trained NNs into one. There is
    # a parameter involved in doing this, alpha, so we try various values of alpha
    # and measure performance.

    # alpha should be between 0 and 1, to interpolate between the 2 datasets
    # we try all possible (?) values between 0 and 1 and assume there are only ever 2 tasks!

    start = alphaStartStopStep[0]
    end = alphaStartStopStep[1]
    step = alphaStartStopStep[2]

    for alpha in np.arange(start, end + step, step):
      alpha_list = [(1 - alpha), alpha]

      if mean_imm: # Mean-IMM
        LW = model_utils.UpdateMultiTaskLwWithAlphas(L_copy[0], alpha_list, no_of_task)
        model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)

        acc = mlp.Test(sess, 'test3', debug=False)
        print('alpha={:.3f} ({}), acc {} MEAN_IMM'.format(alpha, optimizer, acc))
        if plotfile3 is not None: plotfile3.write('{},{}\n'.format(alpha, acc))

      if mode_imm: # Mode-IMM
        LW = model_utils.UpdateMultiTaskWeightWithAlphas(FM, alpha_list, no_of_task)
        model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)

        acc = mlp.Test(sess, 'test3', debug=False)
        print('alpha={:.3f} ({}), acc {} MODE_IMM'.format(alpha, optimizer, acc))
        if plotfile3 is not None: plotfile3.write('{},{}\n'.format(alpha, acc))

    # for
    print('Used Time: {:.4f}s'.format(time.time() - start_time))


if __name__ == '__main__':

    parser = defaultParser.create_default_parser()
    # additional model hyperparameters
    parser.add_argument('--regularizer', type=float,
                        default=0.01,
                        help="L2 Regularization parameter. If > 0, perform l2-IMM, else wtIMM")

    parser.add_argument('--learning_rate2', type=float,
                        default=0.01,
                        help="learning rate on D2, ignored for now..")

    parser.add_argument('--tasks', type=int,
                        default=2,
                        help="number of tasks, if 0: baseline, otherwise it should be 2")

    parser.add_argument('--mean_imm', type=bool,
                        default=True,
                        help='include Mean-IMM')
    parser.add_argument('--mode_imm', type=bool,
                        default=True,
                        help='include Mode-IMM')

    # Model Hyperparameter
    parser.add_argument('--alphaStartStopStep', type=float, nargs=3,
                        default=[0, 1, 0.01],
                        help="alpha(K) of Mean & Mode IMM (cf. equation (3)~(8) in the article)")

    # Training Hyperparameter
    parser.add_argument('--optimizer', type=str,
                        default='SGD', # default in imm is Adam
                        help='''the method name of optimization.
                         * SGD
                         * Adam
                         * Momentum
                        ''')

    # utils.SetDefaultAsNatural(FLAGS)

    FLAGS, unparsed = parser.parse_known_args()
    defaultParser.printFlags(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

