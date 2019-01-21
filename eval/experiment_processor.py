'''
Provide different evaluation strategies (realistic, prescient, ...) 
  usual strategy:
    1. load all experiments (csv-files) from disk (argument: path) (Experiment class is used)
    2. select evaluation strategy (e.g. realistic_first)
      2a. measurement function on D1 is used to find the best hyper-parameter (each per dataset)
      2b. based on the result matrix from 2a the experiments were rebuild with doExperiments to
          retrain the best on D2
          
Created on 30.05.2018

@author: BPF
'''

import argparse
import os
import sys
import time

from eval.experiment import Experiment as Ex
from eval.experiment_plotter import Experiment_Plotter as Ex_Pl # for plotting experiments
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

FLAGS = None

def arg_parser():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--evalMode', type=str,
                      default='imm_search',
                      choices=[
                      'realistic',
                      'prescient',
                      'realistic_first',
                      'realistic_second',
                      'imm_search',
                      'imm_prescient',
                      ],
                      help='''evaluationMethods:
                      realistic (best D1, search best on D2),
                      prescient (search for best D1D2),
                      realistic_first (search only for best D1), 
                      realistic_second (search only for best D1)
                      imm_prescient search for best accuracy for all alpha values (after merge)
                      imm_search search for best on D1, search for best on D2 (variable learn rate), plot D1, D2 and alpha values
                      ''')
  
  parser.add_argument('--path', type=str,
                      default='.\wtIMM', # for IMM use imm_search as evalMode
                      help='path to *.csv files')
  
  parser.add_argument('--task', type=str,
                      default=None,
                      help='if not None, only a given tasks will be processed')
  
  parser.add_argument('--modelID', type=str,
                      default=None,
                      help='if not None, only a given model will be processed'
                      )
  
  parser.add_argument('--dataset', type=str,
                      default=None,
                      help='if not None, only a given dataset will be processed')
  
  return parser


def time_measurement(fn, **params):
  print('start {} with param[s] {}: '.format(fn.__name__, params), end='', flush=True)
  start = time.time()
  fn(**params)
  end = time.time()
  print('{} finished: {:.02f}s'.format(fn.__name__, end - start))


def measure_quality_DAll(exp):
  """  search for 99.9% of the best performance from D2DAll
  
  i = when reach D2D2 k M the maximum?
  quality = performance on whole data set at the time i
  but: read performance from D1 from file _D2D-1
  structure of data_for_model is always (D1D1, D2D2, D2D1, D2DAll)
  
  @param exp: the current experiment object  
  """
  
  D2DAll = exp.data['D2DAll']
  exp._set_result('resultMatrixRetrain', D2DAll[:, 1].max() * 0.999)


def measure_quality_on_D2_099(exp):
  """ search for the best performance on D2D2 and this value for identifying 
  the first iteration step with 99% of this value and return, based on the iteration,
  the performance on D2D1. 
  
  TODO: not on D2D1 but weighted on D2D2[iteration] and weighted on D2D1[iteration]
  
  evaluates based on D2 performance only
  stop criterion is 99.9% of maximal D2 test performance while retraining
  i = when reach D2D2 k M the maximum?
  quality = performance on whole dataset at the time i
  structure of data_for_model is always assumed to be (D1D1, D2D2, D2D1, D2DAll)
  
  @param exp: the current experiment object
  """
  D2D2 = exp.data['D2D2']
  D2D1 = exp.data['D2DAll'] 
  
  maxD2D2 = D2D2[:, 1].max() * 0.999 # define upper limit for search
  for i in range(0, D2D2.shape[0]):# TODO: fix
    if D2D2[i, 1] >= maxD2D2:
      exp._set_result('resultMatrixRetrain', exp.weight1 * D2D1[i, 1] + exp.weight2 * D2D2[i, 1]) 


def measure_quality_on_D1(exp):
  ''' store the best performance on D1 (D1D1) in training result matrix  
  
  @param exp: the current experiment object
  '''
  exp._set_result('resultMatrixTrain', exp.data['D1D1'][:, 1].max())


def measure_quality_on_D2(exp):
  ''' store the best performance on D2 (D2D2) in retraining result matrix
    
  @param exp: the current experiment object
  '''
  exp._set_result('resultMatrixRetrain', exp.data['D2D2'][:, 1].max()) 

def measure_quality_IMM(exp):
  ''' store the best performance varied alpha (D2DAll csv-file)
  
  write mode to resultMatrixTrain
  write mean to resultMatrixTrain
    
  @param exp: the current experiment object
  '''
  exp._set_result('resultMatrixTrain', exp.data['D2DAll'][1::2][:, 1].max()) # mode
  exp._set_result('resultMatrixRetrain', exp.data['D2DAll'][::2][:, 1].max()) # mean
  

def sequentiell_processing(measure_functions):
  ''' apply the measurement function for all experiments 
  
  @param measure_function: function with experiment for parameter (function(Experiment)); is parameter for experiment
  '''
  for exp in Ex.expDict.values(): exp.process_data(measure_functions)


def realistic_evaluation_first_run():
  ''' realistic evaluation method only for D1 
  create retrain experiments with best parameters on D1 with doExperiemts.py 
  '''
  # process all read in experiments with given measurement function (results in resultMatrixTrain)
  time_measurement(sequentiell_processing, measure_functions=measure_quality_on_D1)
  
  # load resultMatrixTrain with results from sequentiell_processing 
  result_matrix_train = Ex.get_result_matrix('resultMatrixTrain')
  
  # search for the best params on each task and dataset
  best_exp_on_tasks = result_matrix_train.argmax(axis=1)
  
  for index, param_id in np.ndenumerate(best_exp_on_tasks):
    task_id = index[0]
    dataset_id = index[1]
    
    # load best experiment on all tasks 
    exp = Ex.matrix_lookup(task_id, param_id, dataset_id)
    
    # create retrain experiment 
    exp.create_retrain_experiments()


def realistic_evaluation_second_run(): 
  ''' realistic evaluation method only for D2 
  
  create plots for best experiments on D2
  
  outputs:
    * plot every single experiment (best)
    * plot lightweight variant of every single experiment (best), commented out
    * recreate_experiment for second processing, commented out
    
    * plot all similar experiments in one 3d plot (all datasets)
    * summarize all best tasks for D5-5, D9-1, DP10-10; search for worst experiment; print latex table column, commented out  
  '''
  # process all read in experiments with given measurement function (results in resultMatrixRetrain)
  time_measurement(sequentiell_processing, measure_functions=measure_quality_on_D2)
  
  # load resultMatrixRetrain 
  result_matrix_retrain = Ex.get_result_matrix('resultMatrixRetrain')
  
  # search for the best params on each task and dataset
  best_exp_on_tasks = result_matrix_retrain.argmax(axis=1) 
  
  # contain all (best) experiments for one task
  all_exp_task = {}
  
  for index, param_id in np.ndenumerate(best_exp_on_tasks):
    task_id = index[0]
    dataset_id = index[1]
    
    # load best experiment on all tasks 
    exp = Ex.matrix_lookup(task_id, param_id, dataset_id)
    
    # load all other experiments with same fixed parameters
    similar_experiments = Ex.resolve_fix_parameter_string(exp.fix_params)
    
    # search for best on D2 with different fixed parameters (default: ['relr'])
    best = max(similar_experiments, key=lambda exp: exp.load_result_value('resultMatrixRetrain'))
    
    # append best experiment to corresponding task_id list (for joint plot) 
    if task_id not in all_exp_task: all_exp_task[task_id] = []
    all_exp_task[task_id].append(best)
    
    # print('Experiment({}) Performance: D1({}) D2({})'.format(exp.params, exp.load_result_value('resultMatrixTrain'), best))
    best.plot()                                                          # 2D plot for single experiment
    # best.plot_light()                                                  # lightweight 2D plot 
    # best.recreate_experiment()                                         # recreate experiment with doExperiments.py
    
  # create group plot for comparable experiment types (different datasets)
  time_measurement(Ex_Pl.plot_all_exp, experiments=all_exp_task)         # 3D plot with all datasets
  # time_measurement(Ex_Pl.create_LaTeX_table, experiments=all_exp_task) # LaTeX table column
    

def realistic_evaluation():
  ''' realistic evaluation method 
  
  @deprecated: use realistic_evaluation_first_run and realistic_evaluation_second_run. (will be deleted in further versions)
   
  evaluate all experiments in a single step if training D1 and retraining on D2 is done for all parameter at once.
  Needs more computational resources compared to process only D1 and afterwords D2
  '''
  # process data with measurement functions
  time_measurement(sequentiell_processing, measure_functions=measure_quality_on_D1)
  time_measurement(sequentiell_processing, measure_functions=measure_quality_on_D2)
  
  # load resultMatrixTrain 
  result_matrix_train = Ex.get_result_matrix('resultMatrixTrain')
  
  # search for the best params on each task and dataset
  best_exp_on_tasks = result_matrix_train.argmax(axis=1) 
  
  # contain all (best) experiments for one task
  all_exp_task = {}
  
  for index, param_id in np.ndenumerate(best_exp_on_tasks):
    task_id = index[0]
    dataset_id = index[1]
    
    # load best experiment on all tasks 
    exp = Ex.matrix_lookup(task_id, param_id, dataset_id)
    
    # load all other experiments with same fixed parameters
    similar_experiments = Ex.resolve_fix_parameter_string(exp.fix_params)
    
    # search for best on D2 with different fixed parameters (default: ['relr'])
    best = max(similar_experiments, key=lambda exp: exp.load_result_value('resultMatrixRetrain'))
    
    # append best experiment to corresponding task_id list (for joint plot) 
    task_id = best.task
    if task_id not in all_exp_task: all_exp_task[task_id] = []
    all_exp_task[task_id].append(best)
    
    # print('Experiment({}) Performance: D1({}) D2({})'.format(exp.params, exp.load_result_value('resultMatrixTrain'), best))
    best.plot()
    
  time_measurement(Ex_Pl.plot_all_exp, experiments=all_exp_task)  
 

def prescient_evaluation():
  ''' prescient evaluation method (not used anymore)
  
  @deprecated: not further investigated. (will be deleted in further versions)
  '''
  time_measurement(sequentiell_processing, measure_functions=measure_quality_DAll)
  
  # load resultMatrixRetrain 
  result_matrix_retrain = Ex.get_result_matrix('resultMatrixRetrain')
  
  # search for the best params on each task and dataset
  best_exp_on_tasks = result_matrix_retrain.argmax(axis=1) 
  
  for index, param_id in np.ndenumerate(best_exp_on_tasks):
    task_id = index[0]
    dataset_id = index[1]
    
    # load best experiment on all tasks 
    exp = Ex.matrix_lookup(task_id, param_id, dataset_id)
    # print('Experiment({}) Performance: DAll({})'.format(exp.params, exp.load_result_value('resultMatrixRetrain')))

def imm_search(_mode):
  ''' imm evaluation method 
  
  search for best experiment depending on alpha value of mode and mean variant
  
  outputs:
    * plot every single imm experiment (best)
    * plot all similar experiments in one 3d plot (all datasets)
    * summarize all best tasks for D5-5, D9-1; search for worst experiment; print latex table rows, commented out
  '''
  # process data with measurement functions
  time_measurement(sequentiell_processing, measure_functions=measure_quality_on_D1)
  time_measurement(sequentiell_processing, measure_functions=measure_quality_on_D2) 
  
  # load resultMatrixTrain 
  result_matrix_train = Ex.get_result_matrix('resultMatrixTrain')     
  
  # search for the best params on each task and dataset
  best_exp_on_tasks = result_matrix_train.argmax(axis=1)
  
  all_exp_task = {} # contain all (best) experiments for one task
  for index, param_id in np.ndenumerate(best_exp_on_tasks):
    task_id = index[0]
    dataset_id = index[1]
    
    # load best experiment on all tasks 
    exp = Ex.matrix_lookup(task_id, param_id, dataset_id)
    print('exp with best parameters for D1:', exp.params)
    
    print('exp.fix_params', exp.fix_params)
    
    # load all other experiments with same fixed parameters
    similar_experiments = Ex.resolve_fix_parameter_string(exp.fix_params)
    print('similar experiments: ',  [_exp.params for _exp in similar_experiments])
    
    # search for best on D2 with different fixed parameters (default: ['relr'])
    best = max(similar_experiments, key=lambda exp: exp.load_result_value('resultMatrixTrain'))
    
    # append best experiment to corresponding task_id list (for joint plot) 
    if task_id not in all_exp_task: all_exp_task[task_id] = []
    all_exp_task[task_id].append(best)
    
    # print('Experiment({}) Performance: D1({}) D2({})'.format(exp.params, exp.load_result_value('resultMatrixTrain'), best))
    Ex_Pl.plot_imm_exp(exp, _mode)
  
  time_measurement(Ex_Pl.plot_all_exp_imm, experiments=all_exp_task, _mode=_mode)         # create group plot for comparable experiment types (different datasets)
  # time_measurement(Ex_Pl.create_LaTeX_table_imm, experiments=all_exp_task, _mode=_mode) # create LaTeX table
  
  
def imm_search_prescient():
  ''' imm evaluation method 
  
  search for best experiment depending on alpha value of mode and mean variant
  
  outputs:
    * plot every single imm experiment (best)
    * plot all similar experiments in one 3d plot (all datasets)
    * summarize all best tasks for D5-5, D9-1; search for worst experiment; print latex table rows, commented out
  '''
  # process data with measurement functions
  time_measurement(sequentiell_processing, measure_functions=measure_quality_IMM) 
  
  # load resultMatrixTrain 
  result_matrix_train = Ex.get_result_matrix('resultMatrixTrain')     # mode
  result_matrix_retrain = Ex.get_result_matrix('resultMatrixRetrain') # mean
  
  # search for the best params on each task and dataset
  best_exp_on_tasks_mode = result_matrix_train.argmax(axis=1)
  best_exp_on_tasks_mean = result_matrix_retrain.argmax(axis=1) 
  
  #---------------------------------------------------------------------------------------------------------------- MODE
  all_exp_task = {} # contain all (best) experiments for one task
  for index, param_id in np.ndenumerate(best_exp_on_tasks_mode):
    task_id = index[0]
    dataset_id = index[1]
    
    # load best experiment on all tasks 
    exp = Ex.matrix_lookup(task_id, param_id, dataset_id)
    
    # load all other experiments with same fixed parameters
    similar_experiments = Ex.resolve_fix_parameter_string(exp.fix_params)
    
    # search for best on D2 with different fixed parameters (default: ['relr'])
    best = max(similar_experiments, key=lambda exp: exp.load_result_value('resultMatrixTrain'))
    
    # append best experiment to corresponding task_id list (for joint plot) 
    if task_id not in all_exp_task: all_exp_task[task_id] = []
    all_exp_task[task_id].append(best)
    
    # print('Experiment({}) Performance: D1({}) D2({})'.format(exp.params, exp.load_result_value('resultMatrixTrain'), best))
    Ex_Pl.plot_imm_exp(exp)
  
  time_measurement(Ex_Pl.plot_all_exp_imm, experiments=all_exp_task, _mode='mode') # create group plot for comparable experiment types (different datasets)
  time_measurement(Ex_Pl.create_LaTeX_table_imm, experiments=all_exp_task, _mode='mode')
  
  #---------------------------------------------------------------------------------------------------------------- MEAN
  all_exp_task = {} # contain all (best) experiments for one task
  
  for index, param_id in np.ndenumerate(best_exp_on_tasks_mean):
    task_id = index[0]
    dataset_id = index[1]
    
    # load best experiment on all tasks 
    exp = Ex.matrix_lookup(task_id, param_id, dataset_id)
    
    # load all other experiments with same fixed parameters
    similar_experiments = Ex.resolve_fix_parameter_string(exp.fix_params)
    
    # search for best on D2 with different fixed parameters (default: ['relr'])
    best = max(similar_experiments, key=lambda exp: exp.load_result_value('resultMatrixRetrain'))
    
    # append best experiment to corresponding task_id list (for joint plot) 
    if task_id not in all_exp_task: all_exp_task[task_id] = []
    all_exp_task[task_id].append(best)
    
    # print('Experiment({}) Performance: D1({}) D2({})'.format(exp.params, exp.load_result_value('resultMatrixTrain'), best))
    Ex_Pl.plot_imm_exp(exp)
    
  # create group plot for comparable experiment types (different datasets) 
  time_measurement(Ex_Pl.plot_all_exp_imm, experiments=all_exp_task, _mode='mean')
  time_measurement(Ex_Pl.create_LaTeX_table_imm, experiments=all_exp_task, _mode='mean')
  

def evaluation(evalMode):
  ''' call the evaluation function '''
  if   evalMode == 'realistic':        realistic_evaluation()
  elif evalMode == 'prescient':        prescient_evaluation()
  elif evalMode == 'realistic_first':  realistic_evaluation_first_run()
  elif evalMode == 'realistic_second': realistic_evaluation_second_run()
  elif evalMode == 'imm_search':       
    imm_search('mode')
    imm_search('mean')
  elif evalMode == 'imm_prescient':    imm_search_prescient()


def check_files(path, model, dataset_name):
  ''' validate experiments, number of experiments for each task and dataset '''

  ds = {}
  ds_sizes = []
  
  #------------------------------------------------------------------------------------------------ evaluate experiments
  for exp_str in Ex.expDict.keys():
    split = exp_str.split('_')
    _task = split[2]
    _lr = split[4]
    _relr = split[6]
    _h1 = split[8]
    _h2 = split[10]
    _h3 = split[12]
    _ds = split[14]
    
    if _ds not in ds: 
      ds[_ds] = {'task':set(),
                 'lr':  set(),
                 'relr':set(),
                 'h1':  set(),
                 'h2':  set(),
                 'h3':  set()}
    
    ds[_ds]['task'].add(_task)
    ds[_ds]['lr'].add(_lr)
    ds[_ds]['relr'].add(_relr)
    ds[_ds]['h1'].add(_h1)
    ds[_ds]['h2'].add(_h2)
    ds[_ds]['h3'].add(_h3)
    
    def add_val(_param):
      if _param not in ds[_ds]: ds[_ds][_param] = 1
      else: ds[_ds][_param] += 1
    
    add_val(_task)
    add_val(_lr)
    add_val(_relr)
    add_val(_h1)
    add_val(_h2)
    add_val(_h3)
  
  #------------------------------------------------------------------------------------------------------- check results
  for _ds, v in sorted(ds.items()):
    sum_tasks = 0
    task_sizes = []
    # check all tasks have same number of experiments for a given dataset
    for _task in v['task']: 
      sum_tasks += v[_task]
      task_sizes.append(v[_task])
    ds_sizes.append(sum_tasks)
    
    if True or len(set(task_sizes)) != 1: 
      print('dataset {}'.format(_ds))
      print('task:', end='')
      for _task in v['task']: print('\t{}'.format(_task), end='')
      print()
      for _task in v['task']: print('\t{}'.format(v[_task]), end='')
      print('\t= {}\n'.format(sum_tasks))
        
    #-------------------------------------------------------------------------------------------------------------------
    def print_param(param):
      if not True: return 
      print('{}:'.format(param), end='')
      if param.startswith('h'):
        if '0' not in v[param]: print('\t\t', end='')
      for _param in sorted(v[param]): print('\t{:5} : {}'.format(_param, v[_param]), end='')
      print()
      
  if len(set(ds_sizes)) != 1: 
    print('not all datasets have the same number of experiments')
    for i, (_ds, v) in enumerate(sorted(ds.items())): 
      print('dataset: {:<10} \t {:>5}'.format(_ds, ds_sizes[i]))
      print_param('lr')
      print_param('relr')
      print_param('h1')
      print_param('h2')
      print_param('h3')
  else:
    print('all datasets have the same number of experiments')
    
  
def main():
  parser = arg_parser()
  global FLAGS
  FLAGS, _ = parser.parse_known_args()
  
  # read experiments from disk
  time_measurement(Ex.build_experiments,
                   path=FLAGS.path,
                   model=FLAGS.modelID,
                   dataset_name=FLAGS.dataset) 
  
  
  
  # check number of experiments
  time_measurement(check_files,
                   path=FLAGS.path,
                   model=FLAGS.modelID,
                   dataset_name=FLAGS.dataset)
  
  # create result matrices
  Ex.create_matrix('resultMatrixTrain')
  Ex.create_matrix('resultMatrixRetrain')
  
  time_measurement(evaluation, evalMode=FLAGS.evalMode)

if __name__ == '__main__':
  main()
   
