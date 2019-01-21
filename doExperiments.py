'''
generates bash files for doing experiments
each experiments produces exactly 4 outputs:
modelID_runID_D1D1.csv, D2D1.csv, D2D2.csv, D2D-1.csv
D1D1: training on D1, test on D1
D2D1: training on D2, test on D1
D2D2: training on D2, test on D2
D2D-1: training on D2, test on D1uD2 using all readout layers. This should be identical to test
'''

import argparse
import itertools
import os
import re

''' OS VARIABLES:
  will set during experiment distribution
    e.g. task will computed on linux machine
      %TMP:ExpDist% will be substituted with "/tmp/ExpDist"
         on windows (different substitution via ssh)
      %TMP:/tmp/ExpDist% will be substituted with "%Temp%\\ExpDist"

  %TMP%         directory for temporary files (created at start, deleted at end) should be on local machine with read and write access
  %SHARE%       shared directory e.g. NFS-Share (Home-Directory) with read and write access
  %LOG%         directory for logs with read and write access
  %INTERPRETER% path/command for interpreter (linux: python3, windows: python)
  %PATH%        path to experiment e.g. ~/IncrementalLearning or %HOMEPATH%\\IncrementalLearning
'''



if False: # create os independent batch file for our experiment distribution system
  PATH = '%PATH%/IncrementalLearning/'
  INTERPRETER = '%INTERPRETER% '
  DEFAULT = INTERPRETER + PATH
  MAX_STEPS = 4500 # TODO: deprecated, will be removed in further versions
  RESULT_PATH = '%TMP%/ExpDist'
  CHECKPOINT_DIR = RESULT_PATH + '/checkpoints/'
else: # create batch file for local execution of experiments
  PATH= "./"
  INTERPRETER="python3 "
  DEFAULT = INTERPRETER + PATH
  RESULT_PATH="./"
  CHECKPOINT_DIR = RESULT_PATH + '/checkpoints/'
  MAX_STEPS = 4500 # TODO: deprecated, will be removed in further versions


DEFAULT_PARAMS = ' --dataset_file {dataset_file} \
                   --hidden1 {hidden1} \
                   --hidden2 {hidden2} \
                   --hidden3 {hidden3} \
                   --max_steps {max_steps} \
                   --batch_size 100 \
                   --dropout_hidden {dropout_hidden} \
                   --dropout_input {dropout_input} \
                   --permuteTrain {permuteTrain} \
                   --permuteTrain2 {permuteTrain2} \
                   --permuteTest {permuteTest} \
                   --permuteTest2 {permuteTest2} \
                   --permuteTest3 {permuteTest3} \
                   --checkpoints_dir {checkpoints_dir} '

cmdTemplates = {
    'wtIMM': {
        'baseline': DEFAULT + 'imm_code/imm.py \
                    --learning_rate {learning_rate} \
                    --learning_rate2 {learning_rate2} \
                    --mean_imm False \
                    --mode_imm False \
                    --optimizer SGD \
                    --plot_file {resultPath}/{model_name}_baseline.csv \
                    --tasks 0 \
                    --train_classes {D1} \
                    --train2_classes {D2} \
                    --test_classes {D1} \
                    --test2_classes {D2} \
                    --mergeTestInto 2 1 \
                    --mergeTrainInto 2 1 ' + DEFAULT_PARAMS,
        'D1D1': DEFAULT + 'imm_code/imm.py \
                    --alphaStartStopStep 0.0 1.0 0.01 \
                    --learning_rate {learning_rate} \
                    --learning_rate2 {learning_rate2} \
                    --mode_imm True \
                    --mean_imm True \
                    --optimizer SGD \
                    --plot_file {resultPath}/{model_name}_D1D1.csv \
                    --train_classes {D1} \
                    --test_classes {D1} \
                    --tasks 1 '  + DEFAULT_PARAMS,
        'D2DAll': DEFAULT + 'imm_code/imm.py \
                  --alphaStartStopStep 0.0 1.0 0.01 \
                  --learning_rate {learning_rate} \
                  --learning_rate2 {learning_rate2} \
                  --mean_imm True \
                  --mergeTestInto 2 3 \
                  --mode_imm True \
                  --optimizer SGD \
                  --plot_file {resultPath}/{model_name}_D1D1.csv \
                  --plot_file2 {resultPath}/{model_name}_D2D2.csv \
                  --plot_file3 {resultPath}/{model_name}_D2DAll.csv \
                  --regularizer 0.0 \
                  --tasks 2 \
                  --train_classes {D1} \
                  --train2_classes {D2} \
                  --test_classes {D1} \
                  --test2_classes {D2} \
                  --test3_classes {D1} ' + DEFAULT_PARAMS
    }, # wtIMM
  'l2tIMM': {
        'baseline': DEFAULT + 'imm_code/imm.py \
                    --learning_rate {learning_rate} \
                    --learning_rate2 {learning_rate2} \
                    --mean_imm False \
                    --mode_imm False \
                    --optimizer SGD \
                    --plot_file {resultPath}/{model_name}_baseline.csv \
                    --tasks 0 \
                    --train_classes {D1} \
                    --train2_classes {D2} \
                    --test_classes {D1} \
                    --test2_classes {D2} \
                    --mergeTestInto 2 1 \
                    --mergeTrainInto 2 1 ' + DEFAULT_PARAMS,
        'D1D1': '', 
        'D2DAll': DEFAULT + 'imm_code/imm.py \
                  --alphaStartStopStep 0.0 1.0 0.01 \
                  --learning_rate {learning_rate} \
                  --learning_rate2 {learning_rate2} \
                  --mean_imm True \
                  --mergeTestInto 2 3 \
                  --mode_imm True \
                  --optimizer SGD \
                  --plot_file {resultPath}/{model_name}_D1D1.csv \
                  --plot_file2 {resultPath}/{model_name}_D2D2.csv \
                  --plot_file3 {resultPath}/{model_name}_D2DAll.csv \
                  --regularizer 0.01 \
                  --tasks 2 \
                  --train_classes {D1} \
                  --train2_classes {D2} \
                  --test_classes {D1} \
                  --test2_classes {D2} \
                  --test3_classes {D1} ' + DEFAULT_PARAMS
    }, # l2IMM

  'fc': {
    'baseline': DEFAULT + 'dnn_code/dnn.py  \
                --learning_rate={learning_rate} \
                --mergeTrainInto 2 1 \
                --mergeTestInto 2 1 \
                --plot_file {resultPath}/{model_name}_baseline.csv \
                --start_at_step 0 \
                --test_classes {D1} \
                --test2_classes {D2} \
                --train_classes {D1} \
                --train2_classes {D2} ' + DEFAULT_PARAMS,
    'D1D1': DEFAULT + 'dnn_code/dnn.py  \
            --learning_rate={learning_rate} \
            --plot_file {resultPath}/{model_name}_D1D1.csv \
            --save_model {model_name}_D1D1 \
            --start_at_step 0 \
            --test_classes {D1} \
            --train_classes {D1} ' + DEFAULT_PARAMS,
    'D2DAll': DEFAULT + 'dnn_code/dnn.py \
              --learning_rate={learning_rate2} \
              --load_model {model_name}_D1D1 \
              --mergeTestInto 2 3 \
              --plot_file  {resultPath}/{model_name}_D2D1.csv \
              --plot_file2 {resultPath}/{model_name}_D2D2.csv  \
              --plot_file3 {resultPath}/{model_name}_D2DAll.csv \
              --start_at_step {start_at_step} \
              --test2_classes {D2} \
              --test3_classes {D1}  \
              --test_classes {D1} \
              --train_classes {D2} ' + DEFAULT_PARAMS
  }, # fc

  'LWTA-fc': {
    'baseline': DEFAULT + 'dnn_code/dnn.py  \
                --dnn_model lwta \
                --learning_rate={learning_rate} \
                --mergeTrainInto 2 1 \
                --mergeTestInto 2 1 \
                --plot_file {resultPath}/{model_name}_baseline.csv \
                --start_at_step 0 \
                --test_classes {D1} \
                --test2_classes {D2} \
                --train_classes {D1} \
                --train2_classes {D2} ' + DEFAULT_PARAMS,
    'D1D1': DEFAULT + 'dnn_code/dnn.py  \
            --dnn_model lwta \
            --learning_rate={learning_rate} \
            --plot_file {resultPath}/{model_name}_D1D1.csv \
            --save_model {model_name}_D1D1 \
            --start_at_step 0 \
            --test_classes {D1} \
            --train_classes {D1} ' + DEFAULT_PARAMS,
    'D2DAll': DEFAULT + 'dnn_code/dnn.py  \
              --dnn_model lwta \
              --learning_rate={learning_rate2} \
              --load_model {model_name}_D1D1 \
              --mergeTestInto 2 3 \
              --plot_file  {resultPath}/{model_name}_D2D1.csv \
              --plot_file2 {resultPath}/{model_name}_D2D2.csv  \
              --plot_file3 {resultPath}/{model_name}_D2DAll.csv \
              --start_at_step {start_at_step} \
              --test2_classes {D2} \
              --test3_classes {D1}  \
              --test_classes {D1} \
              --train_classes {D2} ' + DEFAULT_PARAMS
  }, # LWTA-fc

  'conv': {
    'baseline': DEFAULT + 'dnn_code/dnn.py \
                --dnn_model cnn \
                --learning_rate={learning_rate} \
                --mergeTrainInto 2 1 \
                --mergeTestInto 2 1 \
                --plot_file {resultPath}/{model_name}_baseline.csv \
                --start_at_step 0 \
                --test_classes {D1} \
                --test2_classes {D2} \
                --train_classes {D1} \
                --train2_classes {D2} ' + DEFAULT_PARAMS,
    'D1D1': DEFAULT + 'dnn_code/dnn.py \
            --dnn_model cnn \
            --learning_rate={learning_rate} \
            --plot_file {resultPath}/{model_name}_D1D1.csv \
            --save_model {model_name}_D1D1 \
            --start_at_step 0 \
            --test_classes {D1} \
            --train_classes {D1} ' + DEFAULT_PARAMS,
    'D2DAll': DEFAULT + 'dnn_code/dnn.py \
              --dnn_model cnn \
              --learning_rate={learning_rate2} \
              --load_model {model_name}_D1D1 \
              --mergeTestInto 2 3 \
              --plot_file  {resultPath}/{model_name}_D2D1.csv \
              --plot_file2 {resultPath}/{model_name}_D2D2.csv  \
              --plot_file3 {resultPath}/{model_name}_D2DAll.csv \
              --start_at_step {start_at_step} \
              --test2_classes {D2} \
              --test3_classes {D1}  \
              --test_classes {D1} \
              --train_classes {D2} ' + DEFAULT_PARAMS
  }, # conv

  'EWC': { # TODO: added plot_file2 and 3 to D1D1??? Ok?
    'baseline': DEFAULT + 'ewc_code/ewc_with_options.py \
                --learning_rate={learning_rate} \
                --mergeTrainInto 2 1 \
                --mergeTestInto 2 1 \
                --plot_file {resultPath}/{model_name}_baseline.csv \
                --start_at_step 0 \
                --test_classes {D1} \
                --test2_classes {D2} \
                --train_classes {D1} \
                --train2_classes {D2} ' + DEFAULT_PARAMS,
    'D1D1': DEFAULT + 'ewc_code/ewc_with_options.py \
                --learning_rate={learning_rate} \
                --plot_file {resultPath}/{model_name}_D1D1.csv \
                --save_model {model_name}_D1D1 \
                --start_at_step 0 \
                --test_classes {D1} \
                --train_classes {D1} ' + DEFAULT_PARAMS,
    'D2DAll': DEFAULT + 'ewc_code/ewc_with_options.py \
              --learning_rate={learning_rate2} \
              --load_model {model_name}_D1D1 \
              --mergeTestInto 2 3 \
              --plot_file  {resultPath}/{model_name}_D2D1.csv \
              --plot_file2 {resultPath}/{model_name}_D2D2.csv  \
              --plot_file3 {resultPath}/{model_name}_D2DAll.csv \
              --start_at_step {start_at_step} \
              --test2_classes {D2} \
              --test3_classes {D1}  \
              --test_classes {D1} \
              --train_classes {D2} ' + DEFAULT_PARAMS
  } # EWC
}

permutationSettingsDP = {
  'baseline': {
    'permuteTrain':   '0',
    'permuteTrain2':  '1',
    'permuteTest':    '0',
    'permuteTest2':   '1',
    'permuteTest3':   '0',
    'mergeTrainInto': '2 1',
    'mergeTestInto':  '2 1',
  },
  'D1D1': {
    'permuteTrain':  '0',
    'permuteTrain2': '0',
    'permuteTest':   '0',
    'permuteTest2':  '0',
    'permuteTest3':  '0',
  },
  'D2DAll': {
    'permuteTrain':  '1',
    'permuteTrain2': '1',
    'permuteTest':   '0',
    'permuteTest2':  '1',
    'permuteTest3':  '0',
    'mergeTestInto': '2 3',
  }
}


def generateTaskString(task):
  ''' generate the partition tast strings based on task for D1 (initial training), D2 (retraining) and combination (D1 + D2)
  e.g. D5-5a ->
       D1: 0 1 2 3 4
       D2: 5 6 7 8 9

  @param task: task (str)
  @return: D1 (str), D2 (str)
  '''

  #===========================================================================
  # D5-5
  #===========================================================================
  if task == 'D5-5a':
    D1 = '0 1 2 3 4'
    D2 = '5 6 7 8 9'
  elif task == 'D5-5b':
    D1 = '0 2 4 6 8'
    D2 = '1 3 5 7 9'
  elif task == 'D5-5c':
    D1 = '3 4 6 8 9'
    D2 = '0 1 2 5 7'
  elif task == 'D5-5d':
    D1 = '0 2 5 6 7'
    D2 = '1 3 4 8 9'
  elif task == 'D5-5e':
    D1 = '0 1 3 4 5'
    D2 = '2 6 7 8 9'
  elif task == 'D5-5f':
    D1 = '0 3 4 8 9'
    D2 = '1 2 5 6 7'
  elif task == 'D5-5g':
    D1 = '0 5 6 7 8'
    D2 = '1 2 3 4 9'
  elif task == 'D5-5h':
    D1 = '0 2 3 6 8'
    D2 = '1 4 5 7 9'

  #===========================================================================
  # D9-1
  #===========================================================================
  elif task == 'D9-1a':
    D1 = '0 1 2 3 4 5 6 7 8'
    D2 = '9'
  elif task == 'D9-1b':
    D1 = '1 2 3 4 5 6 7 8 9'
    D2 = '0'
  elif task == 'D9-1c':
    D1 = '0 2 3 4 5 6 7 8 9'
    D2 = '1'

  #===========================================================================
  # DP10-10
  #===========================================================================
  elif task == 'DP10-10':
    D1 = '0 1 2 3 4 5 6 7 8 9'
    D2 = '0 1 2 3 4 5 6 7 8 9'

  return D1, D2

def generateUniqueId(expID, params):
  '''
  takes an experimental ID (see getScriptName) and combines it with the paramaters
  to obtain an unique ID for an experiment 'expID' with parameters 'params'
  '''
  task = params[0]
  learning_rate = params[1]
  retrain_learning_rate = params[2]
  h1 = params[3]
  h2 = params[4]
  h3 = 0
  dataset = params[5]

  if len(params) is 7:
    h3 = params[5]
    dataset = params[6]

  dataset = dataset.split('.')[0]

  uid = '{expID}_task_{task}_lr_{learning_rate}_relr_{retrain_learning_rate}_h1_{h1}_h2_{h2}_h3_{h3}_ds_{dataset}'.format(
    expID=expID,
    learning_rate=learning_rate,
    retrain_learning_rate=retrain_learning_rate,
    h1=h1,
    h2=h2,
    h3=h3,
    dataset=dataset,
    task=task)

  return uid


def generateNewCommandLine(expID, scriptName, resultPath, action, params, maxSteps=2000):
  lrate = str(params[1])
  lrate2 = str(params[2])
  hidden1 = str(params[3])
  hidden2 = str(params[4])
  dataset_file = str(params[5])
  hidden3 = '-1'

  # create layer conf parameters
  if len(params) == 7:
    hidden3 = str(params[5])
    dataset_file = str(params[6])

  D1, D2 = generateTaskString(params[0])

  model_name = scriptName

  # handle dropout parameters
  dropout_input = 1.0
  dropout_hidden = 1.0
  if expID.find('D-') != -1 or expID in ['wtIMM', 'l2tIMM']:
    dropout_input = 0.8
    dropout_hidden = 0.5
    expID = expID.replace('D-', '')

  # remove D- from expID
  canonicalExpID = expID.replace('D-', '')
  execStr = cmdTemplates[canonicalExpID][action]

  # handle permutation settings
  permutationDict = {
    'permuteTrain':  '0',
    'permuteTrain2': '0',
    'permuteTest':   '0',
    'permuteTest2':  '0',
    'permuteTest3':  '0'
  }

  if expID.find('DP') != -1:   # a permuted task such as DP10-10
    permutationDict = permutationSettingsDP[action]

  if execStr == '': return ''

  if expID in ['wtIMM', 'l2tIMM']:
    execStr = execStr.format(D1=D1,
                             D2=D2,
                             addPermutation=False,
                             checkpoints_dir=CHECKPOINT_DIR,
                             dataset_file=dataset_file,
                             dropout_hidden=dropout_hidden,
                             dropout_input=dropout_input,
                             expID=expID,
                             hidden1=hidden1,
                             hidden2=hidden2,
                             hidden3=hidden3,
                             learning_rate=lrate,
                             learning_rate2=lrate2,
                             max_steps=maxSteps,
                             model_name=model_name,
                             plotFile=resultPath + model_name + '_' + action + '.csv',
                             resultPath=resultPath,
                             scriptName=scriptName,
                             start_at_step=maxSteps,
                             **permutationDict)
        
  elif canonicalExpID in ['LWTA-fc', 'fc', 'conv', 'EWC']:
    try: execStr = execStr.format(D1=D1,
                                  D2=D2,
                                  addPermutation=False,
                                  checkpoints_dir=CHECKPOINT_DIR,
                                  dataset_file=dataset_file,
                                  dropout_hidden=dropout_hidden,
                                  dropout_input=dropout_input,
                                  expID=expID,
                                  hidden1=hidden1,
                                  hidden2=hidden2,
                                  hidden3=hidden3,
                                  learning_rate=lrate,
                                  learning_rate2=lrate2,
                                  max_steps=maxSteps,
                                  model_name=model_name,
                                  resultPath=resultPath,
                                  scriptName=scriptName,
                                  start_at_step=maxSteps,
                                  **permutationDict)
    except Exception as ex:
      print(permutationDict)
      raise(ex)
  
  if not FLAGS.fix_params:
    # remove save model string if only performance on D1D1 is tested -> no checkpoint is created
    execStr = re.sub('--save_model.*_D1D1 ', ' ', execStr) 
    
  execStr = re.sub('\s+', ' ', execStr) + '\n' # strip
  return execStr 

def generate_download_regex(uniqueID):
  uniqueID.replace('.', '\.')
  return 'DOWNLOAD:{}_.*\.csv\n'.format(uniqueID)

def generate_delete_regex(uniqueID):
  uniqueID.replace('.', '\.')
  return 'DELETE:{}/{}_*ckpt*\n'.format(CHECKPOINT_DIR, uniqueID)

def generate_experiment_options(uniqueID):
  options = {}

  # TODO: currently not used
  # TODO: calculate GPU Memory by number of layers and sizes, and dnn type (factor) for an optimized experiment distribution
  # TODO: load values from specific dataset class
  if 'FashionMNIST' in uniqueID: gpu_mem = (28 * 28 * 1) * (60000 + 10000)
  elif 'notMNIST' in uniqueID:   gpu_mem = (28 * 28 * 1) * (529114 + 18724)
  elif 'EMNIST' in uniqueID:     gpu_mem = (28 * 28 * 1) * (697932 + 116323)
  elif 'MNIST' in uniqueID:      gpu_mem = (28 * 28 * 1) * (55000 + 10000)
  elif 'CIFAR10' in uniqueID:    gpu_mem = (32 * 32 * 3) * (50000 + 10000)
  elif 'MADBase' in uniqueID:    gpu_mem = (28 * 28 * 1) * (60000 + 10000)
  elif 'Devangari' in uniqueID:  gpu_mem = (32 * 32 * 1) * (82800 + 9200)
  elif 'SVHN' in uniqueID:       gpu_mem = (32 * 32 * 3) * (73257 + 26032)
  elif 'Simpsons' in uniqueID:   gpu_mem = (100 * 100 * 3) * (6789 + 754)
  elif 'CUB200' in uniqueID:     gpu_mem = (100 * 100 * 3) * (70000 + 10000)
  elif 'Fruits' in uniqueID:     gpu_mem = (100 * 100 * 3) * (32426 + 10903)
  else: gpu_mem = -1

  if FLAGS.exp == 'conv':
    max_other_processes = 1
    gpu_mem = -1 # need whole gpu memory TODO: test
  else: max_other_processes = -1

  options['GM'] = gpu_mem
  options['MoP'] = max_other_processes
  options['RAM'] = -1
  options['CPU'] = -1

  return 'OPTIONS:' + str(options) + '\n'


def validParams(t):
  ''' filter h1 and h2 == 0 parameter combinations
   
  @param t: parameter tuple
  @return: False if h1 or h2 is 0, else True
  '''
  _, _, _, h1, h2, _, _ = t
  return not (h1 == 0 or h2 == 0)


def correctParams(t):
  task, lrTrain, lrRetrain, h1, h2, h3, datasets = t
  if h3 == 0: return (task, lrTrain, lrRetrain, h1, h2, datasets)
  else: return t


def arg_parser():
  ''' create a parser '''
  parser = argparse.ArgumentParser()

  parser.add_argument('--exp',
                      default='wtIMM',
                      choices=[
                      'fc', 'D-fc',
                      'LWTA-fc',
                      'conv', 'D-conv',
                      'EWC', 'D-EWC',
                      'wtIMM', 'l2tIMM'
                      ],
                      help=''' Tasks:
                      fc, D-fc
                      LWTA-fc
                      conv, D-conv
                      EWC, D-EWC
                      wtIMM, l2tIMM
                      ''')

  parser.add_argument('--num_files', type=int,
                      default=1,
                      help='Number of files the experiment is divided into')
  
  parser.add_argument('--fix_params', type=str,
                      default=None,
                      help='recreate whole experiment for best D1, if "None" create all without fix params e.g. for IMM')
  
  parser.add_argument('--recreate', type=str,
                      default='None',
                      help='recreate whole experiment')
  
  return parser


parser = arg_parser()
FLAGS, unparsed = parser.parse_known_args()

tasks = [ 'D5-5a', 'D5-5b', 'D5-5c', 'D5-5d', 'D5-5e', 'D5-5f', 'D5-5g', 'D5-5h', 'D9-1a', 'D9-1b', 'D9-1c', 'DP10-10' ]

train_learning_rates = [0.01, 0.001]
retrain_learning_rates = [0.001]

if FLAGS.fix_params: 
  retrain_learning_rates = [0.001, 0.0001, 0.00001]

if FLAGS.exp.find('conv') != -1: layerSizes = [1]
else: layerSizes = [0, 200, 400, 800]

parser = arg_parser()
FLAGS, unparsed = parser.parse_known_args()

tasks = [ 'D5-5a', 'D5-5b', 'D5-5c', 'D5-5d', 'D5-5e', 'D5-5f', 'D5-5g', 'D5-5h', 'D9-1a', 'D9-1b', 'D9-1c', 'DP10-10' ]

train_learning_rates = [0.01, 0.001]
retrain_learning_rates = [0.001] # not used for first determination of D1
datasets = [ 'MNIST.pkl.gz',
             'FashionMNIST.pkl.gz',
             'NotMNIST.pkl.gz',
             'CIFAR10.pkl.gz',
             'MADBase.pkl.gz',
             # 'CUB200.pkl.gz', # only 60 images per class
             'Devnagari.pkl.gz',
             'EMNIST.pkl.gz',
             'Fruits.pkl.gz',
             # 'Simpsons.pkl.gz',
             'SVHN.pkl.gz' ]

if FLAGS.fix_params and FLAGS.fix_params != 'None':
  retrain_learning_rates = [0.001, 0.0001, 0.00001]
  fields = FLAGS.fix_params.split('_')
  print(fields)
  experiment_type = fields[0]
  FLAGS.exp = experiment_type
  tasks = fields[2]
  learning_rate = float(fields[4])
  # retrain_learning_rate = float(fields[6]) # substituted
  h1 = int(fields[8])
  h2 = int(fields[10])
  h3 = int(fields[12])
  dataset = fields[14]
  
  combinations = itertools.product([tasks], # experiment_type
                                   [learning_rate],
                                   retrain_learning_rates,
                                   [h1],
                                   [h2],
                                   [h3],
                                   [dataset])
  
  validCombinations = [correctParams(t) for t in combinations if validParams(t)]
  
elif FLAGS.recreate and FLAGS.recreate != 'None': # TODO: bugfix for recreate best experiments in dnn.py
  print('recreate', FLAGS.recreate)
  fields = FLAGS.recreate.split('_')
  print('fields', fields)
  experiment_type = fields[0]
  FLAGS.exp = experiment_type
  tasks = fields[2]
  learning_rate = float(fields[4])
  retrain_learning_rate = float(fields[6])
  h1 = int(fields[8])
  h2 = int(fields[10])
  h3 = int(fields[12])
  dataset = fields[14]
  
  combinations = itertools.product([tasks], # experiment_type
                                   [learning_rate],
                                   retrain_learning_rates,
                                   [h1],
                                   [h2],
                                   [h3],
                                   [dataset])
  
  validCombinations = [correctParams(t) for t in combinations if validParams(t)]
else: # no params fixed (initial run)
  print('no fix params')
  if FLAGS.exp.find('conv') != -1: layerSizes = [1]
  else: layerSizes = [0, 200, 400, 800]
  
  combinations = itertools.product(tasks,
                                   train_learning_rates,
                                   retrain_learning_rates,
                                   layerSizes,
                                   layerSizes,
                                   layerSizes,
                                   datasets)
  
  validCombinations = [correctParams(t) for t in combinations if validParams(t)]

limit = 40000
n = 0
index = 0
files = [ open(FLAGS.exp + '-part-' + str(n) + '.bash', 'w' if not FLAGS.fix_params else 'a') for n in range(0, int(FLAGS.num_files)) ]

for task in validCombinations:
  uniqueID = generateUniqueId(FLAGS.exp, task)

  f = files[n]
  f.write('--- START ---\n')
  f.write(generateNewCommandLine(FLAGS.exp, uniqueID, RESULT_PATH, 'D1D1', task, maxSteps=MAX_STEPS))       # initial training
  if FLAGS.fix_params:
    f.write(generateNewCommandLine(FLAGS.exp, uniqueID, RESULT_PATH, 'baseline', task, maxSteps=MAX_STEPS)) # initial training
    f.write(generateNewCommandLine(FLAGS.exp, uniqueID, RESULT_PATH, 'D2DAll', task, maxSteps=MAX_STEPS))   # retraining and evaluation on D1
  f.write(generate_download_regex(uniqueID))     # append: "DOWNLOAD:D-fc_task_D5-5a_lr_0.001_relr_0.001_h1_200_h2_200_h3_0_ds_NotMNIST_.*\.csv" (regex)
  f.write(generate_experiment_options(uniqueID)) # append: "OPTIONS:{'GM':1000,'MoP':-1,'RAM':-1, 'CPU':-1}
  f.write(generate_delete_regex(uniqueID))       # append: "DELETE:%TMP%/ExpDist/checkpoints/fc_task_D5-5a_lr_0.01_relr_0.001_h1_200_h2_200_h3_0_ds_NotMNIST*.ckpt.*" (no regex!)
  f.write('--- END ---\n')

  n += 1
  if n >= int(FLAGS.num_files): n = 0
  index += 1
  if index >= limit: break

print('created {} experiments for exp: {}'.format(len(validCombinations), FLAGS.exp))

for f in files: f.close()
