'''
Created on 06.06.2018

@author: BPF
'''

from dataset.data_loader import Data_Loader as DL
from dataset.dataset import Dataset as DS
from dataset.downloader import Downloader as DWL


class Devnagari(DS):
  ''' The Devnagari Character Dataset (DHCD) handwritten digits

    source: https://www.kaggle.com/rishianand/devanagari-character-set
    
    website: https://web.archive.org/web/20160105230017/http://cvresearchnepal.com/wordpress/dhcd/
    training set of 78200 images (32 x 32 grey scale) for 46 characters classes (1700 images per class)
    test set of 13800 images (32 x 32 grey scale) for 46 characters classes (300 images per class)
    
    @INPROCEEDINGS{7400041, 
        author={S. Acharya and A. K. Pant and P. K. Gyawali}, 
        booktitle={2015 9th International Conference on Software, Knowledge, Information Management and Applications (SKIMA)}, 
        title={Deep learning based large scale handwritten Devanagari character recognition}, 
        year={2015}, 
        volume={}, 
        number={}, 
        pages={1-6}, 
        keywords={document image processing;handwritten character recognition;learning (artificial intelligence);neural nets;optical character recognition;DHCD;Devanagari handwritten character dataset;Devanagari script;dataset increment approach;deep CNN;deep convolutional neural network;deep learning architecture;dropout approach;handwritten documents;large scale handwritten Devanagari character recognition;optical character recognition;public image dataset;Character recognition;Convolution;Kernel;Neural networks;Testing;Training;Computer Vision;Deep Convolutional Neural Network;Deep learning;Devanagari Handwritten Character Dataset;Dropout;Image processing;Optical Character Recognition}, 
        doi={10.1109/SKIMA.2015.7400041}, 
        ISSN={}, 
        month={Dec},
        }
  '''

  def __init__(self):
    super(Devnagari, self).__init__()

  @property
  def name(self): return 'Devanagari Character Dataset'
  
  @property
  def shape(self): return (32, 32, 1)

  @property
  def pickle_file(self): return 'Devanagari.pkl.gz'

  @property
  def num_train(self): return 82800

  @property
  def num_test(self): return 9200

  @property
  def num_classes(self): return 10 # 46
  
  def download_data(self, force=False): 
    self.compressed_file = DWL.download_from_kaggle('rishianand/devanagari-character-set', 'Images.zip', force=force)

  def extract_data(self, force=False):
    self.extracted_files = DL.extract(self.compressed_file, FORCE=force)
      
  def prepare_data(self): 
    self.data_train, self.labels_train = DL.load_directory_as_dataset(self.extracted_files + '/Images/', self.shape, max_classes=self.num_classes)
    
    # Only True, if 10% (default) of data split in to training and testing
    self.validate_shape = False
  
  def clean_up(self):
    DS.file_delete('Images.zip')
    DS.file_delete('Images/')

  
if __name__ == '__main__':
  Devnagari().get_data(clean_up=False)
