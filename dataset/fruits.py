'''
Created on 06.06.2018

@author: BPF
'''

import os

from dataset.data_loader import Data_Loader as DL
from dataset.dataset import Dataset as DS
from dataset.downloader import Downloader as DWL


class Fruits(DS):
  ''' Fruits 360 dataset: A dataset of images containing fruits
  
  source: https://www.kaggle.com/moltean/fruits
  (they change often the file format)
  
  website: https://github.com/Horea94/Fruit-Images-Dataset
  Training 34641 images (100 x 100 color) Number of classes: 69 (fruits). 
  Test 11640 images (100 x 100 color) Number of classes: 69 (fruits).
  
  @article{DBLP:journals/corr/abs-1712-00580,
    author    = {Horea Muresan and
                 Mihai Oltean},
    title     = {Fruit recognition from images using deep learning},
    journal   = {CoRR},
    volume    = {abs/1712.00580},
    year      = {2017},
    url       = {http://arxiv.org/abs/1712.00580},
    archivePrefix = {arXiv},
    eprint    = {1712.00580},
    timestamp = {Wed, 03 Jan 2018 12:33:17 +0100},
    biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1712-00580},
    bibsource = {dblp computer science bibliography, https://dblp.org}
  }
  '''

  def __init__(self):
      super(Fruits, self).__init__()
  
  @property
  def name(self): return 'Fruits Dataset'
  
  @property
  def shape(self): return (100, 100, 3)

  @property
  def pickle_file(self): return 'Fruits.pkl.gz'

  @property
  def num_train(self): return 32426

  @property
  def num_test(self): return 10903

  @property                        # TODO: MemoryError! in interp1d: split?
  def num_classes(self): return 10 # 65 TODO: fix zip bug or split pickle files
  
  def download_data(self, force=False): 
    self.compressed_file = DWL.download_from_kaggle('moltean/fruits', 'fruits.zip', force=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)

    # Extract contained zip
    # sub_file = os.listdir(self.extracted_files)
    # print('sub_file', sub_file)
    # if len(sub_file) != 1: raise Exception('invalid kaggle file format (more than one file in zip)')
    
    # print('self.extracted_files before', self.extracted_files)
    # self.extracted_files = DL.extract(os.path.join(self.extracted_files, sub_file[0]), FORCE=force)
    # print('self.extracted_files', self.extracted_files)
      
  def prepare_data(self):
    self.data_train, self.labels_train, self.data_test, self.labels_test = DL.load_directory_as_dataset(self.extracted_files + '/fruits-360/Training/',
                                                                                                        self.shape,
                                                                                                        self.extracted_files + '/fruits-360/Test/',
                                                                                                        max_classes=self.num_classes)
    
    # False, if not all classes were used
    self.validate_shape = False
      
  def clean_up(self):
    DS.file_delete('fruits.zip')
    DS.file_delete('fruits/')


if __name__ == '__main__':
  Fruits().get_data(clean_up=False)
