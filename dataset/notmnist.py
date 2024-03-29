'''
Created on 06.06.2018

@author: BPF
'''
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from dataset.data_loader import Data_Loader as DL
from dataset.dataset import Dataset as DS
from dataset.downloader import Downloader as DWL


class NotMNIST(DS):
  ''' notMNIST dataset (current source contains invalid images?)
  
  website: http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html
  other possible source: https://www.kaggle.com/jwjohnson314/notmnist
  
  train 529114 (28 x 28 gray scale) images
  test 18724 (28 x 28 gray scale) images
  '''

  def __init__(self):
    super(NotMNIST, self).__init__()
  
  @property
  def name(self): return 'NotMNIST'
  
  @property
  def shape(self): return (28, 28, 1)

  @property
  def pickle_file(self): return 'NotMNIST.pkl.gz'

  @property
  def num_train(self): return 529114

  @property
  def num_test(self): return 18724

  @property
  def num_classes(self): return 10
  
  def download_data(self, force=False): 
    self.compressed_file_training = DWL.download_file('http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz', FORCE=force)
    self.compressed_file_test = DWL.download_file('http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz', FORCE=force)

  def extract_data(self, force=False):
    self.extracted_files_training = DL.extract(self.compressed_file_training, FORCE=force)
    self.extracted_files_test = DL.extract(self.compressed_file_test, FORCE=force)
      
  def prepare_data(self):
    self.data_train, self.labels_train, self.data_test, self.labels_test = DL.load_directory_as_dataset(self.extracted_files_training + '/notMNIST_large/',
                                                                                                        self.shape,
                                                                                                        self.extracted_files_test + '/notMNIST_small/')
  
  def clean_up(self):
    DS.file_delete('notMNIST_large/')
    DS.file_delete('notMNIST_small/')


if __name__ == '__main__':
  NotMNIST().get_data(clean_up=False)
