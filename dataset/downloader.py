'''
Created on 06.06.2018

@author: BPF
'''
import kaggle
import os
import urllib


class Downloader(object):
  ''' Provice download functions for files by HTTP(S) and Kaggle-API '''
  
  DATASET_DIR = './datasets/'
  DOWNLOAD_DIR = DATASET_DIR + 'download/'

  def __init__(self, params): pass
      
  @classmethod
  def download_from_kaggle(cls, dataset, filename=None, path=None, force=False, quiet=False):
    ''' download a dataset file from kaggle 
    
    You need a kaggle account!
    '''

    if os.path.isfile('~/.kaggle/kaggle.json'):
      raise Exception('''no kaggle account? 
      1. create a kaggle account (https://www.kaggle.com/)
      2. on your profile (My Account) -> "Create New API Token"
      3. save kaggle.json in ~/.kaggle/
      4. enjoy
      ''') 

    if path is None: path = cls.DOWNLOAD_DIR
    if filename is not None and os.path.isfile(path + filename) and not force: return path + filename

    kaggle.api.dataset_download_cli(dataset, file_name=filename, path=path, force=force, quiet=quiet)

    if filename is None: return path
    return path + filename
  
  @classmethod
  def download_file(cls, target_url, file_name=None, FORCE=False):
    ''' download a file from target_url

    @param target_url: target_url
    @param target_dir: target directoy (str)
    @param file_name: target filename (str)
    @param FORCE: if True (bool): overwrite existing file
    @return: path to output directory (str)
    '''
    if file_name is None:
      file_name = target_url.split('/')[-1]
      print('use filename {} from url'.format(file_name))

    output_dir_file = os.path.join(cls.DOWNLOAD_DIR, file_name)

    print('download from {} to {} ({})'.format(target_url, output_dir_file, 'FORCE' if FORCE else ''))

    last_percent = -1

    def call_back_download(count, block_size, total_size):
      nonlocal last_percent
      percent = int(count / (total_size / block_size) * 100)
      if percent > last_percent: print('Download {}% file {} '.format(percent, output_dir_file,))
      last_percent = percent

    if not os.path.isdir(cls.DOWNLOAD_DIR):
      os.makedirs(cls.DOWNLOAD_DIR)

    if os.path.isfile(output_dir_file) and not FORCE:
      print('file {} already exists'.format(output_dir_file))
      return output_dir_file
    else:
      urllib.request.urlretrieve(target_url, output_dir_file, call_back_download)
      return output_dir_file
