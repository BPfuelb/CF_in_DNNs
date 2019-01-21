'''
Code base for EWC experiments.

@author: BPF (only modified) 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from dataset.dataset import load_dataset
import defaultParser
from ewc_code.classifiers import Classifier
import tensorflow as tf

FLAGS = None


def train(dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3, properties):
  ''' ... '''
  sess = tf.InteractiveSession() # Start an Interactive session 

  classifier = Classifier(num_class=properties['num_classes'],
                          num_features=reduce(lambda x, y: x * y, properties['dimensions']) * properties['num_of_channels'],
                          fc_hidden_units=[FLAGS.hidden1 , FLAGS.hidden2] if FLAGS.hidden3 == -1 else [FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3],
                          apply_dropout=True,
                          checkpoint_path=FLAGS.checkpoints_dir,
                          flags=FLAGS,
                          dataSetTrain=dataSetTrain)
      
  print('\nTraining on DataSet started...\n____________________________________________________________')

  testdatalist = [dataSetTest, dataSetTest2, dataSetTest3] 
  if not FLAGS.test2_classes and not FLAGS.test3_classes: testdatalist = [dataSetTest]
     
  plot_files = [file for file in [FLAGS.plot_file, FLAGS.plot_file2, FLAGS.plot_file3] if file is not None]
  
  classifier.train_mod(sess=sess,
                       model_name=FLAGS.save_model,         # if None...no checkpoints created
                       model_init_name=FLAGS.load_model,
                       dataset=dataSetTrain,
                       fisher_multiplier=1.0 / FLAGS.learning_rate,
                       learning_rate=FLAGS.learning_rate,
                       testing_data_sets=testdatalist,
                       plot_files=plot_files)


def main(_):
  defaultParser.printFlags(FLAGS)
  
  dataSetTrain, _, dataSetTest, dataSetTest2, dataSetTest3, properties = load_dataset(FLAGS)
  train(dataSetTrain, dataSetTest, dataSetTest2, dataSetTest3, properties)


if __name__ == '__main__':
  parser = defaultParser.create_default_parser()
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
