'''
Created on 06.06.2018

@author: BPF
'''

from abc import ABC
from abc import abstractmethod
from abc import abstractproperty
import argparse
from collections import namedtuple
from functools import reduce
import importlib
import os
from scipy.interpolate import interp1d
import shutil
import sys
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet as TF_DataSet
import time

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from dataset.data_loader import Data_Loader as DL
import defaultParser
import numpy as np


class Dataset(ABC):
  ''' Abstract Base Class for all datasets (template pattern)

  define:
      abstract properties (setter and getter): name, shape, pickle_file, num_train, num_test, num_classes
                                               data_train, labels_train, data_test, labels_test
      abstract methods for derived classes: download_data, prepare_data, extract_data
      template for dataset construction: get_data
  '''
  
  DATASET_DIR = './datasets/'
  DOWNLOAD_DIR = DATASET_DIR + 'download/'

  Range = namedtuple('Range', 'min max')
  VALUE_RANGE = Range(-1.0, +1.0) # range of data values mapped e.g. from [0, 255] to [-1.0, +1.0]
  SPLIT_TRAINING_TEST = 10        # % if only training data exists, split training to: 90% training, 10% testing

  def __init__(self):
    ''' define internal data structure for mapping  '''
    self._data_train = None
    self._labels_train = None
    self._data_test = None
    self._labels_test = None
    self._validate_shape = True

  def __str__(self): 
    ''' string representation of a dataset '''
    s = ''
    s += 'Dataset: {}\n'.format(self.name)
    s += '  * pickle file name: {}\n'.format(self.pickle_file)
    s += '  * shape: {}\n'.format(self.shape)
    s += '  * number train examples: {}\n'.format(self.num_train)
    s += '  * number test examples: {}\n'.format(self.num_test)
    s += '  * num_classes: {}\n'.format(self.num_classes)
    s += '  * validate shape: {}\n'.format(self._validate_shape)
    return s
    
  def __repr__(self): return self.__str__()

  #----------------------------------------------------------------------------------------------------- abstract fields
  @abstractproperty
  def name(self): pass

  @abstractproperty
  def shape(self): pass

  @abstractproperty
  def pickle_file(self): pass

  @abstractproperty
  def num_train(self): pass

  @abstractproperty
  def num_test(self): pass

  @abstractproperty
  def num_classes(self): pass

  #--------------------------------------------------------------------------------------------------- setter and getter
  @property
  def validate_shape(self): return self._validate_shape

  @validate_shape.setter
  def validate_shape(self, value): self._validate_shape = value

  #------------------------------------------------------------------------------
  @property
  def data_train(self): return self._data_train

  @data_train.setter
  def data_train(self, value): self._data_train = value

  #------------------------------------------------------------------------------
  @property
  def labels_train(self): return self._labels_train

  @labels_train.setter
  def labels_train(self, value): self._labels_train = value

  #------------------------------------------------------------------------------
  @property
  def data_test(self): return self._data_test

  @data_test.setter
  def data_test(self, value): self._data_test = value

  #------------------------------------------------------------------------------
  @property
  def labels_test(self): return self._labels_test

  @labels_test.setter
  def labels_test(self, value): self._labels_test = value

  #------------------------------------------------------------------------------
  @property
  def is_regression(self): return False
  #---------------------------------------------------------------------------------------------------------------------- 

  def get_data(self, force=False, clean_up=False):
    ''' We'll fire you if you override this method.

    template method define the general procedure to create a dataset:
        * if: no stored data exists:
            * download data (by dataset)
            * extract data (by dataset) TF_DataSet
            * prepare data (by dataset)
            * convert: split (optional), shuffle, flatten (optional)
            * store data (pickel)
            * load data (pickel)
            * validate data loaded data
            * cleanup (optional) (by dataset)
        * else:
            * load stored data (pickle file)

    @param force: flag (boolean) for download and extract mechanism to overwrite existing files
    @return: data_train (np.array), labels_train (np.array), data_test (np.array), _labels_test ((np.array), properties (dict())
    '''
    print('get_data for {}'.format(self.name))

    if Dataset.__file_exists(self.pickle_file) and not force:
      # TODO: evaluation every time after loading?
      return DL.load_pickle_file(self.pickle_file)

    self.download_data(force)
    self.extract_data(force)
    self.prepare_data()

    # abbreviated form (locals)
    data_train = self.data_train
    labels_train = self.labels_train
    data_test = self.data_test
    labels_test = self.labels_test
    
    # shuffle train
    data_train, labels_train = self.__shuffle_data_and_labels(data_train, labels_train)

    # if no test data exists, split training data (default = 10% test data)
    if self.data_test is None and self.labels_test is None:
      print('split training data into test and training: default = 90% training, 10% test')
      data_train, labels_train, data_test, labels_test = Dataset.__split_train_test(data_train, labels_train, percent_test=Dataset.SPLIT_TRAINING_TEST) # percent_test=10

    # shuffle test
    data_test, labels_test = self.__shuffle_data_and_labels(data_test, labels_test)

    # flatten
    print('data_train', data_train.shape)
    data_train = self.__flatten(data_train)
    print('data_test', data_test.shape)
    data_test = self.__flatten(data_test)

    # normalize if not in range
    data_train = self.__normalize_data(data_train)
    data_test = self.__normalize_data(data_test)

    # to one hot if not one hot
    labels_train = self.__to_one_hot(labels_train, 'train')
    labels_test = self.__to_one_hot(labels_test, 'test')

    # create pickle
    properties = {'num_classes': self.num_classes, 'num_of_channels': self.shape[2], 'dimensions': list(self.shape[0:-1])}
    DL.pickle_data(self.pickle_file, data_train, labels_train, data_test, labels_test, properties)

    if clean_up: self.clean_up()

    # invert abbreviated form
    self.data_train = data_train
    self.labels_train = labels_train
    self.data_test = data_test
    self.labels_test = labels_test

    # load pickle file and validate (overwrite self properties)
    return self.validate()

  @abstractmethod
  def download_data(self): pass

  @abstractmethod
  def prepare_data(self): pass

  @abstractmethod
  def extract_data(self): pass

  @abstractmethod
  def clean_up(self): pass

  def validate(self):
    ''' validate data

    1. load pickle file
    2. check number of elements with num_train and num_test from class (if validate_shape)
    3. check shape of data elements with shape from class
    4. check shape of labels
    5. check types of values (np.float32)
    6. check data values are in range (Dataset.value_range)
    7. check the labels (is to one hot)
    8. check property dict

    @raise Exception: a lot of Exceptions
    '''
    # load data from pickle file
    data_train, labels_train, data_test, labels_test, property = DL.load_pickle_file(self.pickle_file)
    
    def check_all_data_available():
      ''' 1. load pickle files '''
      if data_train is None:   raise Exception('data_train is None')
      if labels_train is None: raise Exception('labels_train is None')
      if data_test is None:    raise Exception('data_test is None')
      if labels_test is None:  raise Exception('labels_test is None')
    
    def check_data_shape():
      ''' 2. check number of dataset elements (data, lables) (if validate_shape)'''
      if self.validate_shape:
        if data_train.shape[0] != self.num_train:   raise Exception('data_train({}) != {}'.format(data_train.shape[0], self.num_train))
        if labels_train.shape[0] != self.num_train: raise Exception('labels_train({}) != {}'.format(labels_train.shape[0], self.num_train))
        if data_test.shape[0] != self.num_test:     raise Exception('data_test({}) != {}'.format(data_test.shape[0], self.num_test))
        if labels_test.shape[0] != self.num_test:   raise Exception('labels_test({}) != {}'.format(labels_test.shape[0], self.num_test))

    def check_data_element_shape():
      ''' 3. check shape of elements (number of values) 
      TODO: if validate_shape: how about variable datasets? e.g. change number of classes? add Flag?
      '''
      data_train_shape = (reduce(lambda x, y: x * y, self.shape),)
      if data_train.shape[1:] != data_train_shape: raise Exception('data_train.shape({}) != {}'.format(data_train.shape, data_train_shape))
  
      data_test_shape = (reduce(lambda x, y: x * y, self.shape),)
      if data_test.shape[1:] != data_test_shape: raise Exception('data_test.shape({}) != {}'.format(data_test.shape, data_test_shape))
      
    def check_labels_element_shape():
      ''' 4. check shape of labels '''
      if self.is_regression:
        pass # TODO: add check for regression labels
      else:  # classification problem
        if labels_train.shape[1] != self.num_classes: raise Exception('labels_train.shape({}) != {}'.format(labels_train.shape[1], self.num_classes))
        if labels_test.shape[1] != self.num_classes:  raise Exception('labels_test.shape({}) != {}'.format(labels_test.shape[1], self.num_classes))

    def check_value_types():
      ''' 5. check types of values '''
      if data_train.dtype != np.float32:   raise Exception('data_train type {} != np.float32'.format(data_train.dtype))
      if labels_train.dtype != np.float32: raise Exception('labels_train type {} != np.float32'.format(labels_train.dtype))
      if data_test.dtype != np.float32:    raise Exception('data_test type {} != np.float32'.format(data_test.dtype))
      if labels_test.dtype != np.float32:  raise Exception('labels_test type {} != np.float32'.format(labels_test.dtype))

    def check_value_range():
      ''' 6. check values of data are in range '''
      labels_test_max = labels_test.max()
      labels_test_min = labels_test.min()
      
      if self.is_regression: pass
      else: # classification problem
        if Dataset.VALUE_RANGE.min < labels_test_max > Dataset.VALUE_RANGE.max: raise Exception('max value({}) out of range {}'.format(labels_test_max, Dataset.VALUE_RANGE))
        if Dataset.VALUE_RANGE.min < labels_test_min > Dataset.VALUE_RANGE.max: raise Exception('max value({}) out of range {}'.format(labels_test_max, Dataset.VALUE_RANGE))

    def check_all_labels():
      ''' 7. check the labels '''

      def check_labels(labels, test_or_training):
        if self.is_regression: return# all labels should be numeric!
        labels_min = labels.min()
        if labels_min != 0: raise Exception('the min({}) should be 0 in labels_{}'.format(labels_min, test_or_training))
        labels_max = labels.max()
        if labels_max != 1: raise Exception('the max({}) should be 1 in labels_{}'.format(labels_max, test_or_training))
        labels_sum = labels.sum()
        if labels_sum != labels.shape[0]: raise Exception('the sum({}) should be same as elements({}) labels_{}'.format(labels_sum, labels.shape[0], test_or_training))

      check_labels(labels_train, 'training')
      check_labels(labels_test, 'test')

    def check_property_dict():
      ''' 8. check property dict '''
      num_classes = property['num_classes']
      if self.is_regression:
        if num_classes != 1: raise Exception('property["num_classes"]({}) != 1 {} (regression problem)'.format(num_classes, self.num_classes))
      else:
        if num_classes != labels_train.shape[1]: raise Exception('property["num_classes"]({}) != self.num_classes {}'.format(num_classes, self.num_classes))
  
      num_of_channels = property['num_of_channels']
      if num_of_channels != self.shape[-1]: raise Exception('property["num_of_channels"]({}) != self.num_classes {}'.format(num_of_channels, self.shape[-1]))
  
      dimensions = property['dimensions']
      if dimensions != list(self.shape[0:-1]): raise Exception('property["num_of_channels"]({}) != self.num_classes {}'.format(num_of_channels, list(self.shape[0:-1])))
    
    check_all_data_available()
    check_data_shape()
    check_data_element_shape()
    check_labels_element_shape()
    check_value_types()
    check_value_range()
    check_all_labels()
    check_property_dict()
    
    def write_log():
      with open('{}.log'.format(self.name), 'w') as log_file:
        log_file.write(self.__str__()) # write all user defined properties
        
        s = ''
        s += ' creation log:\n'
        s += '  * data training shape: {}\n'.format(self.data_train.shape)
        s += '  * labels training shape: {}\n'.format(self.labels_train.shape)
        s += '  * data test shape: {}\n'.format(self.data_test.shape)
        s += '  * labels test shape: {}\n'.format(self.labels_test.shape)
        
        def get_class_distribution(labels):
          c = ''
          class_distribution = np.sum(labels, axis=0)
          class_min, class_max = class_distribution.min(), class_distribution.max()
          
          class_diff = class_max - class_min
          class_diff_percent = class_diff / labels.shape[0] * 100
          c += '   + min({}), max({}), '.format(class_min, class_max)
          c += 'max diff ({}), percent from all ({:.2f}%)\n'.format(class_diff, class_diff_percent)
          for _class, num_elements in np.ndenumerate(class_distribution):
            c += '    - class {}: {}\n'.format(_class[0], num_elements)
          return c
        
        s += '  * elements per class (training):\n{}\n'.format(get_class_distribution(self.labels_train))
        s += '  * elements per class (training):\n{}\n'.format(get_class_distribution(self.labels_test))
        s += ' validation ok...everything seems good'
        
        log_file.write(s)
    
    write_log()
    return data_train, labels_train, data_test, labels_test, property

  def __shuffle_data_and_labels(self, data:np.array, labels:np.array) -> (np.array, np.array):
    ''' shuffle random data and labels

    @param data: input data
    @param labels: input labels
    @raise Exception: invalid shapes

    @return: shuffled data (np.array), shuffled labels (np.array)
    '''
    if data.shape[0] != labels.shape[0]:
      raise Exception('data.shape[0] ({}) != labels.shape[0] ({})'.format(data.shape[0], labels.shape[0]))

    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = data[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

  @classmethod
  def __file_exists(cls, file_or_dir:str) -> bool :
    ''' check if file_or_dir or directory in DATASET_DIR exists

    @param file_or_dir: file or directory (str)
    @return: file or directory exists (bool)
     '''
    path_and_filename = os.path.join(cls.DATASET_DIR, file_or_dir)

    return os.path.isfile(path_and_filename) or os.path.isdir(path_and_filename)

  @classmethod
  def file_delete(cls, file_or_directory:str):
    ''' delete files or directories in DOWNLOAD_DIR

    @param file_or_directory: file or directory (str)
    @raise Exception: if file or directory not found
    '''
    file_or_directory = cls.DOWNLOAD_DIR + file_or_directory

    if os.path.isfile(file_or_directory):
      print('delete: ' + file_or_directory)
      os.remove(file_or_directory)
      return

    if os.path.isdir(file_or_directory):
      print('delete: ' + file_or_directory)
      shutil.rmtree(file_or_directory)
      return

    raise Exception('file or directory not found {}'.format(file_or_directory))

  def __normalize_data(self, data:np.array, value_range:namedtuple=None) -> np.array:
    ''' normalize input data from [0, 255 (pixel_depth)] to e.g. [-1.0, 1.0] (Dataset.VALUE_RANGE)

    @param data: input data (np.array)
    @param value_range: target value range namedtuple('min'(float), 'max'(float))
    @return: normalized array (np.array)
    '''
    # workaround because class properties cannot be used as default parameters
    if value_range is None: value_range = Dataset.VALUE_RANGE

    # print(data.min() , value_range.min , data.max() , value_range.max)
    if data.min() >= value_range.min and data.max() <= value_range.max:
      print("values are in range {} don't do mapping".format(value_range))
      return data

    print('values are not in range!', data.min() , ' and ' , data.max())

    min_range = 0
    max_range = 255

    # if other values are set in subclass, use this
    if hasattr(self, 'min_range') and hasattr(self, 'max_range'):
      min_range = self.min_range
      max_range = self.max_range

    data = interp1d([min_range, max_range], list(value_range))(data) # interpolation
    return data.astype(np.float32)

  def strip_1_dims(self, data:np.array, axis:int=None) -> np.array:
    '''  remove single-dimensional entries
    @param data: data to squeeze (np.array)
    @param axis: if None: search for single-dimensions, else squeeze  axis (int)

    @return: squeezed array (np.array)
    '''
    return np.squeeze(data, axis)

  def __flatten(self, data:np.array):
    ''' flatten inplace the input data from [?, x,y,z,...] to [?, x*y*z*...)]

    @param data: input data (np.array)
    '''
    shape = (data.shape[0], -1)
    data = data.reshape(shape)
    return data

  def __to_one_hot(self, labels:np.array, name='') -> np.array:
    ''' convert input array (labels)
    e.g. from: [0, 3, 1]
         to: [[1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 1, 0, 0]]

    @param labels: input labels (np.array) with shape (?, 1)
    @return: to on hot converted array (np.array)
    '''
    try:
      # check already is one_hot
      self.check_labels(labels, name)
      print('is already one hot, nothing to do')
      return labels
    except:
      # is not one hot...try to convert
      pass

    if len(labels.shape) != 1:
      print('to_one_hot: (len(labels.shape({})) != 1) dont know what to do'.format(labels.shape))
      if labels.shape[1] == 1:
        print('strip 1 dimension and go on with to_one_hot')
        labels = self.strip_1_dims(labels)
      else:
        return labels

    min_label = int(labels.min())
    max_label = int(labels.max())

    # check labels start at 0 and end at num_classes -1, else shift with -1
    if min_label == 1:
      labels = labels - 1
      max_label -= 1
      min_label -= 1

    if min_label != 0: raise Exception('to_one_hot: maximum class should be {} but is {}'.format(self.num_classes, max_label))

    if max_label != self.num_classes - 1: raise Exception('to_one_hot: maximum class should be {} but is {}'.format(self.num_classes - 1, max_label))

    # check all classes are available
    for _class in range(0, self.num_classes - 1):
      if _class not in labels: raise Exception('class ({}) missing in labels'.format(_class))

    # convert to one hot
    to_one_hot = np.zeros([labels.size, self.num_classes])  # allocate the to_one_hot array
    to_one_hot[np.arange(labels.size), labels.astype(np.int32).ravel()] = 1  # convert labels to one_hot

    return to_one_hot.astype(np.float32)

  @classmethod
  def __split_train_test(cls, data:np.array, labels:np.array, percent_test:int) -> (np.array, np.array, np.array, np.array):
    ''' split the input data and labels into training data and labels and test data and labels by a given percentage

    @param data: input data (np.array)
    @param labels: input labels (np.array)
    @param percent_test: percent value (int) of the amount of test data/lables (default = 10%)
    @return: data_train (np.array), data_test (np.array), data_test (np.array), label_test (np.array)
    '''
    num_elements_test = int(len(data) * (percent_test / 100))

    data_test, label_test = data[:num_elements_test], labels[:num_elements_test]
    data_training, labels_training = data[num_elements_test:], labels[num_elements_test:]

    return data_training, labels_training, data_test, label_test


def load_dataset(FLAGS) -> (TF_DataSet, TF_DataSet, TF_DataSet, TF_DataSet, TF_DataSet, dict()):
    ''' load and prepare dataset

     * load dataset based on FLAGS: dataset_dir, dataset_file
     * create permuted data if FLAGS: permuteTrain, permuteTrain2, permuteTest, permuteTest2, permuteTest3
     * exclude/filter classes by FLAGS: train_classes, train2_classes, test_classes, test2_classes, test3_classes
     * merge/combine dataset e.g. permuted and not permuted data; FLAGS: mergeTrainInto, mergeTestInto
     * convert to TensorFlow DataSet objects

    @param FLAGS: parameter used for processing (argparse.Namespace)
    @return: different datasets (e.g. permuted or combinded) (TF_DataSet, TF_DataSet, TF_DataSet, TF_DataSet, TF_DataSet, dict())
     '''

    #===========================================================================
    # Load base dataset
    #===========================================================================
    data_train, labels_train, data_test, labels_test, properties = get_dataset('', '', FLAGS)

    #===========================================================================
    # Permute dataset
    #===========================================================================
    def permute_data(data, seed):
      ''' permute the data if seed (Flag.permuteXXX) != 0 '''
      if seed == 0: return data

      np.random.seed(seed)
      permutation = np.random.permutation(data.shape[1])
      return data[:, permutation].copy()

    # creates permuted COPIES!
    data_train1 = permute_data(data_train, FLAGS.permuteTrain)
    data_train2 = permute_data(data_train, FLAGS.permuteTrain2)
    #------------------------------------------------------------------------------
    data_test1 = permute_data(data_test, FLAGS.permuteTest)
    data_test2 = permute_data(data_test, FLAGS.permuteTest2)
    data_test3 = permute_data(data_test, FLAGS.permuteTest3)

    #===========================================================================
    # Filter classes from data and labels and create DataSet object
    #===========================================================================
    def filter_data_and_labels(classes, data, labels):
      ''' remove classes from dataset by FLAG '''
      if classes and len(classes) != properties['num_classes']:
        # create index mask from labels and apply on data and labels and create TensorFlow DataSet object
        mask = np.isin(labels.argmax(axis=1), classes)
        return TF_DataSet(255. * data[mask], labels[mask], reshape=False)

      return TF_DataSet(255. * data.copy(), labels.copy(), reshape=False)

    train1 = filter_data_and_labels(FLAGS.train_classes, data_train1, labels_train)
    train2 = filter_data_and_labels(FLAGS.train2_classes, data_train2, labels_train)
    #------------------------------------------------------------------------------
    test1 = filter_data_and_labels(FLAGS.test_classes, data_test1, labels_test)
    test2 = filter_data_and_labels(FLAGS.test2_classes, data_test2, labels_test)
    test3 = filter_data_and_labels(FLAGS.test3_classes, data_test3, labels_test)

    #===========================================================================
    # Merge datasets e.g. permuted and not permuted
    #===========================================================================
    def merge(flag, data_labels:[TF_DataSet]):
      ''' merge a dataset into another one !inplace!

      e.g. append not permuted data and labels to permuted data and labels

      choose source (by flag) e.g.      flag[0] = 0 := train1
      choose destination (by flag) e.g. flag[1] = 1 := train2

      append source data/labels on destination data/labels

      shuffle concatenated datasets (again)
      @param flag: flag (argparse.Namespace) should contain a [int] with two element
                   should be in range of data list (data_labels)
      @param data_labels: data to concatenate [TF_DataSet]
      @raise Exception: if flag have invalid parameter (len() != 2, flag[0] == flag[1],)
      '''
      # shift input selection e.g. [2, 1] -> train2 in train1 is [1, 0]

      if flag is None or (type(flag) is int and flag == -1) or flag[0] == -1: return
      if len(flag) != 2 : raise Exception('invalid merge flag [{}]'.format(flag))
      if flag[0] == flag[1] : raise Exception('invalid merge flag [{}]'.format(flag))

      flag = [f - 1 for f in flag]

      if not 0 <= flag[0] < len(data_labels): raise Exception('flag[0] out of range ({})'.format(flag[0]))
      if not 0 <= flag[1] < len(data_labels): raise Exception('flag[1] out of range ({})'.format(flag[1]))

      # select source and destination dataset (data and labels) from data_labels
      src_data = data_labels[flag[0]]
      dst_data = data_labels[flag[1]]

      src_lines = src_data.images.shape[0]
      dst_lines = dst_data.images.shape[0]

      # inplace resize
      dst_data.images.resize([src_lines + dst_lines, src_data.images.shape[1]], refcheck=False)
      dst_data.labels.resize([src_lines + dst_lines, src_data.labels.shape[1]], refcheck=False)

      # append data from source
      dst_data.images[dst_lines:, :] = src_data.images
      dst_data.labels[dst_lines:, :] = src_data.labels

      # shuffle new dataset.. otherwise classes are all in a definite order which would be bad..
      # not 2 be confused with shuffling within images: here, we shuffle just samples
      np.random.seed(0)
      np.random.shuffle(dst_data.images)
      np.random.seed(0)
      np.random.shuffle(dst_data.labels)

    # merge two dataset (e.g. permuted and not permuted) for more data, if flags are set
    merge(FLAGS.mergeTrainInto, [train1, train2])
    merge(FLAGS.mergeTestInto, [test1, test2, test3])

    return train1, train2, test1, test2, test3, properties


def get_dataset(directory:str='', file:str='', FLAGS:argparse.Namespace=None) -> (np.array, np.array, np.array, np.array, dict):
    ''' Load the dataset

    If pickle file exists, load it.
    Else invoke the corresponding class file (based on 'file'), create instance and return .get_data().
        The dataset will be downloaded, converted, pickled and returned (should be prepared before running experiments!).

    @param directory: directory (str) for loading pickle files
    @param file: name (str) of the pickle file
    @param FLAGS: object (Namespace) with parameters form argparser

    @return: data train (np.array), labels train (np.array), data test (np.array), labels test (np.array), properties (dict(str, obj))
    '''

    if not FLAGS is None:
      directory = FLAGS.dataset_dir
      file = FLAGS.dataset_file

    pickle_file = os.path.join(directory, file)

    if os.path.isfile(pickle_file):
      print('\t * try load dataset: {} from {}'.format(file, directory))

      data = DL.load_pickle_file(pickle_file)
      return data

    # remove file ending
    if file.endswith('.pkl.gz'): class_name = file.replace('.pkl.gz', '')
    else: class_name = file

    # import class.file (class file must have the same name as the pickle file!)
    module_class = importlib.import_module('dataset.' + class_name.lower())

    # load class, create instance, return get_data
    return getattr(module_class, class_name)().get_data(clean_up=True)


def main():
    FLAGS = defaultParser()
    start = time.time()
    load_dataset(FLAGS)
    end = time.time()
    print('TIME sum: {:3.2f}s'.format(end - start))


if __name__ == '__main__':
    main()

