'''
Created on 29.05.2018

@author: BPF
'''
import csv
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

print('matplotlib version:', matplotlib.__version__)

class Experiment(object):
  ''' represent a experiment on the basis of the outputs (csv files). Is used for evaluation. '''
  # retraining experiments parameter
  
  # class attributes
  expDict = {}
  _results = {}
  
  # will later converted to dictionaries in build_lookup_tables()
  task_lookup_table = set()    # {task (str) : task_id (int)}
  param_lookup_table = set()   # {params (str): param_id (int)}
  dataset_lookup_table = set() # {dataset (str): dataset_id (int)} 

  inv_task_lookup_table = {}    # {task_id (int): task (str)}
  inv_param_lookup_table = {}   # {param_id (int): params (str)}
  inv_dataset_lookup_table = {} # {dataset_id (int): dataset (str)}
  
  fix_parameters_pattern = {}   # {param (str) : replace_pattern (re.pattern)} 
  fix_parameter_exp = {}
  
  __fails = [] # defect experiments 
  
  MATRIX_LOOKUP_TABLE = 'matrix_lookup_table'
  
  def __init__(self, run_id, exp_id, filename_and_path):
    ''' init a Experiment-object with the first file (run_id, filename and path) and an exp_id '''
    self.__run_id = run_id
    self.__exp_id = exp_id
    self.__files = dict()
    self.__data = dict()
    self._results = dict()
    
    self.__self_check_done = False
    self.__error = ''
    
    self.__assign_to_variables(filename_and_path)
    self.__calc_weights_for_avg()
    self.__add_to_lookup_tables()
  
  def __assign_to_variables(self, filename_and_path):
    ''' assign experiment values to variables
    e.g.: 
         {expID}_task_{task}_lr_{learning_rate}_retrainlr_{retrain_learning_rate}_h1_{h1}_h2_{h2}_h3_{h3}_ds_{dataset}_{action}.csv
        ^-------^----^------^--^---------------^---------^-----------------------^--^----^--^----^--^----^--^---------^--------^
        |   0   |  1 |  2   |3 |      4        |    5    |         6             |7 | 8  |9 | 10 |11| 12 |13|   14    |   15   |
        |                   +-----------------------------------------------------------------------------------------|        |
        |                                                               params                                                 |
        +----------------------------------------------------------------------------------------------------------------------+
                                                                        fields                                            
    '''
    fields, filepath = self.__split_filename(filename_and_path)
    self.__fields = fields
    self.__files[fields[-1]] = filename_and_path
    
    self.__filepath = filepath
    self.__experiment_type = fields[0]
    self.__task = fields[2]
    self.__learning_rate = float(fields[4])
    self.__retrain_learning_rate = float(fields[6])
    self.__h1 = int(fields[8])
    self.__h2 = int(fields[10])
    self.__h3 = int(fields[12])
    self.__dataset = fields[14]
    self.__params = '_'.join(fields[1:-1])
    
    self.__add_to_fix_parameter(self.__params)

  def __split_filename(self, filepath_and_name):
    ''' split the filepath_and_name into:
    
    ./lwta-fc-MRL/LWTA-fc-MRL_task_D5-5_lr_0.001_relr_0.001_h1_200_h2_800_h3_400_ds_MNIST_D2DAll.csv
    ^------------^-----------^----^----^--^-----^----^-----^--^---^--^---^--^---^--^-----^------^
    filepath     |      0    | 1  |  2 |3 | 4   | 5  |  6  |7 | 8 |9 |10 |11|12 |13| 14  |  15  |
                 +------------------------------------------------------------------------------+
                                                    fields            
                                                                          
    @param filepath_and_name: filepath and filename (str) 
    '''
    split_filepath_filename = filepath_and_name.split(os.sep)
    
    filepath = os.sep.join(split_filepath_filename[:-1]) + os.sep
    
    filename = split_filepath_filename[-1]
    filename_no_extension = filename.replace('.csv', '')
    
    fields = filename_no_extension.split('_')
    
    return fields, filepath
  
  def __add_to_fix_parameter(self, params):
    ''' get fix_paramter string add experiment (self) to list of similar experiments '''
    self.__fix_params = Experiment.make_fix_parameter_string(params)
    
    if self.__fix_params not in Experiment.fix_parameter_exp: 
      Experiment.fix_parameter_exp[self.__fix_params] = []
    
    Experiment.fix_parameter_exp[self.__fix_params].append(self)
  
  def __add_to_lookup_tables(self):
    ''' add the task and the params of the experiment to a set of of elements for further transformation to lookup tables '''
    Experiment.task_lookup_table.add(self.__task)
    Experiment.param_lookup_table.add(self.__params)

  def __calc_weights_for_avg(self):
    ''' calculate the weight average of a task (proportionate)

    e.g: task is D9-1c
        9 / (9 + 1) , 1 / (9 + 1)
    '''
    p1 = float(re.search(r'\d+', self.__task.split("-")[0]).group())
    p2 = float(re.search(r'\d+', self.__task.split("-")[1]).group())
    self.__weight1 = p1 / (p1 + p2)
    self.__weight2 = p2 / (p1 + p2)
          
  def __str__(self):
    ''' string representation of an experiment
    TODO: print all file status (ok or not) 
    '''
    s = ''
    s += self.__run_id
    s += '\n  files from filepath {}: '.format(self.__filepath)
    for action, filename in self.__files.items():
      s += '\n    action:{} filename: {}'.format(action, filename)   
    s += '\n  experiment_type: {}'.format(self.__experiment_type)
    s += '\n  task: {} (weights: {}:{})'.format(self.__task, self.__weight1, self.__weight2)
    s += '\n  learning rate: {}'.format(self.__learning_rate)
    s += '\n  retraining learning rate: {}'.format(self.__retrain_learning_rate)
    s += '\n  layers hidden1: {}'.format(self.__h1)
    s += '\n  layers hidden2: {}'.format(self.__h2)
    s += '\n  layers hidden3: {}'.format(self.__h3)
    s += '\n  dataset: {}'.format(self.__dataset)
    s += '\n  params: {}'.format(self.__params)
    s += '\n  fix_params: {}'.format(self.__fix_params)
    # s += '\n  max(D1D1): {}'.format(self._results['resultMatrixTrain'])
    if not self.__ok: s += '\n  error: {}'.format(self.__error)
    return s
  
  def _add_file(self, filepath_and_name):
    ''' add a file to the file list of an experiment, based on the action
    
    @param filepath_and_name: filepath and names (str)
    '''
    
    fields, _ = self.__split_filename(filepath_and_name)
    self.__files[fields[-1]] = filepath_and_name
  
  def _read_files(self):
    ''' read all corresponding files from filesystem and convert csv-files into numpy arrays and store it as {action: array}
    
    np.array dtype.float64 shape: (?:2)
    '''
    try:
      for action, filepath_and_name in self.__files.items():
        with open(filepath_and_name, 'r') as file:
          self.__data[action] = np.array([ [float(val) for val in row] for row in csv.reader(file) if row ])
    except Exception as ex:
      print(ex, filepath_and_name)
    
    return self.__data
  
  def process_data(self, *functions):
    ''' apply all functions with self reference fn(self) '''
    
    if self.__ok: 
      for fn in functions: fn(self)
    else:
      # TODO: do something with the defect experiments
      print('experiment({}) self check failed: {}'.format(self.__exp_id, self.__error))
      pass
  
  def _self_check(self):
    ''' check parameters, data etc. '''
    ok = True
    
    # TODO: check fields 
    
    # check data format
    if len(self.__data) < 1: # set to 1 for first run TODO: build distinction for first and second run issue #20
      self.__append_error('less than 1 actions in data')
      ok = False
    
    def check_dataset(action):
      if action not in self.__data:
        self.__append_error('{} in data missing'.format(action))
        return False
      
      data = self.__data[action] 
      
      if len(data.shape) != 2:
        self.__append_error('{} has wrong shape {} (should (?,?))'.format(action, data.shape))
        return False 
      
      if data.shape[0] < 20:
        self.__append_error('{} has not enough data elements {} (should > 20)'.format(action, data.shape[0]))
        return False
      
      if data.shape[1] != 2:
        self.__append_error('{} has wrong number of values {} (should be 2)'.format(action, data.shape[1]))
        return False
      
      return True

    ok &= check_dataset('D1D1')
    # ok &= check_dataset('D2D2') # disable for first run
    # ok &= check_dataset('D2D1')
    # ok &= check_dataset('baseline')
    # ok &= check_dataset('D2DAll') # disable for first run
    
    self.__self_check_done = True
    if not ok: Experiment.__fails.append(self)
    
    self.__ok = ok # indicates if the experiment should be processed
  
  def __append_error(self, msg):
      self.__error += '{} '.format(msg)
  
  @property
  def data(self):       return self.__data
  
  @property
  def weight1(self):    return self.__weight1
  
  @property
  def exp_id(self):     return self.__exp_id
  
  @property
  def params(self):     return self.__params
  
  @property
  def task(self):       return self.__task
  
  @property
  def fix_params(self): return self.__fix_params
  
  @property
  def weight2(self):    return self.__weight2
  
  @property
  def result(self):     return self._results 
  
  @property
  def dataset(self):    return self.__dataset
    
  @property
  def fields(self):     return self.__fields
      
  def _load_mapping(self):
    ''' use lookup tables to store the own mapping ids for the assignment of the _results 
    e.g. 
         task_id = 1    # 'D9-1b'
         param_id = 342 # 'lr_0.001_relr_0.0001_h1_800_h2_200_h3_800_ds_MNIST'
         dataset_id = 0 # 'MNIST'
    '''
    self.__task_id = Experiment.task_lookup_table[self.__task]
    self.__param_id = Experiment.param_lookup_table[self.__params]
    self.__dataset_id = Experiment.dataset_lookup_table[self.__dataset]

    # add self to matrix_lookup_table
    if self.__ok:        
      Experiment._results[Experiment.MATRIX_LOOKUP_TABLE][self.__task_id, self.__param_id, self.__dataset_id] = self
  
  def create_plot_vertices(self, layer, training='D1D1', retraining=['D2D2', 'D2DAll'], baseline='baseline'):
    ''' generate tuples (triples) of (iteration, layer, performance) of D1D1, D2D2 and D2DAll
    
    @param layer: the y-layer in result plot
    @return: triple of (iteration, layer, performance) of D1D1, D2D2 and D2DAll
     '''
    # prepare D1D1 data
    D1D1_iteration = self.__data[training][:, 0]          # 0, 30, 60, 90, 0, 30 ...
    epochs = np.count_nonzero(D1D1_iteration == 0)        # get number of epochs  (count zero numbers)
    D1D1_iteration = np.arange(0.0, epochs, epochs / D1D1_iteration.shape[0])
    D1D1 = self.__data[training][:, 1]                    # performance values 
    D1D1[0], D1D1[-1] = 0.0, 0.0                          # set start and end to 0
    
    # prepare D2D2 data
    D2D2_iteration = self.__data[retraining[0]][:, 0]     # 0, 30, 60, 90, 0, 30 ...
    epochs = np.count_nonzero(D2D2_iteration == 0)        # get number of epochs  (count zero numbers)
    D2D2_iteration = np.arange(epochs, 2 * epochs, epochs / D2D2_iteration.shape[0])
    D2D2 = self.__data[retraining[0]][:, 1]               # performance values 
    D2D2[0], D2D2[-1] = 0.0, 0.0                          # set start and end to 0
    
    # prepare D2DAll data
    D2DAll_iteration = self.__data[retraining[1]][:, 0]   # 0, 30, 60, 90, 0, 30 ...     
    epochs = np.count_nonzero(D2DAll_iteration == 0)      # get number of epochs  (count zero numbers)
    D2DAll_iteration = np.arange(epochs, 2 * epochs, epochs / D2DAll_iteration.shape[0]) 
    D2DAll = self.__data[retraining[1]][:, 1]             # performance values       
    D2DAll[0], D2DAll[-1] = 0.0, 0.0                      # set start and end to 0   
    
    # prepare baseline data
    basline_height = self.__data[baseline][:, 1].max()    # performance values (max), baseline height       
    basline_width = .5                                    # width of the bar
    baseline_pos_x0 = 2 * epochs                          # start x-position for baseline (bar)
    baseline_pos_x1 = 2 * epochs + basline_width          # end x-position for baseline (bar)
    
    # build triple (iteration, layer, performance)
    D1D1 = list(zip(D1D1_iteration, [layer] * len(D1D1), D1D1))
    D2D2 = list(zip(D2D2_iteration, [layer] * len(D2D2), D2D2))
    D2DAll = list(zip(D2DAll_iteration, [layer] * len(D2DAll), D2DAll))
    baseline = list(zip([baseline_pos_x0, baseline_pos_x0 , baseline_pos_x1 , baseline_pos_x1], [layer] * 4, [0, basline_height, basline_height, 0]))
    
    return [D1D1], [D2D2], [D2DAll], [baseline]

  def create_plot_vertices_imm(self, layer, training='D1D1', retraining=['D2D2', 'D2DAll'], baseline='baseline'):
    ''' generate tuples (triples) of (iteration, layer, performance) of D1D1, D2D2 and D2DAll
    
    @param layer: the y-layer in result plot
    @return: triple of (iteration, layer, performance) of D1D1, D2D2 and D2DAll
     '''
    # prepare D1D1 data
    D1D1_iteration = self.__data[training][:, 0]          # 0, 30, 60, 90, 0, 30 ...
    epochs = np.count_nonzero(D1D1_iteration == 0)        # get number of epochs  (count zero numbers)
    D1D1_iteration = np.arange(0.0, epochs, epochs / D1D1_iteration.shape[0])
    D1D1 = self.__data[training][:, 1]                    # performance values 
    D1D1[0], D1D1[-1] = 0.0, 0.0                          # set start and end to 0
    
    # prepare D2D2 data
    D2D2_iteration = self.__data[retraining[0]][:, 0]     # 0, 30, 60, 90, 0, 30 ...
    epochs = np.count_nonzero(D2D2_iteration == 0)        # get number of epochs  (count zero numbers)
    D2D2_iteration = np.arange(epochs, 2 * epochs, epochs / D2D2_iteration.shape[0])
    D2D2 = self.__data[retraining[0]][:, 1]               # performance values 
    D2D2[0], D2D2[-1] = 0.0, 0.0                          # set start and end to 0
    
    # prepare alpha mode and mean data                                  
    mean = self.__data[retraining[1]][::2]                # all mean imm [alpha,performance] values
    mode = self.__data[retraining[1]][1::2]               # all mode imm [alpha,performance] values
    
    mean_data = mean[:, 1]                                #
    mode_data = mode[:, 1]                                #  
    mean_data[0], mean_data[-1] = 0.0, 0.0                # set start and end to 0
    mode_data[0], mode_data[-1] = 0.0, 0.0                # set start and end to 0
       
    num_alphas = mean_data.shape[0] 
    mean_mode_iteration = np.arange(2 * epochs + 2, 4 * epochs + 2, 2 * epochs / num_alphas)

    # prepare baseline data
    basline_height = self.__data[baseline][:, 1].max()    # performance values (max), baseline height       
    basline_width = 2.0                                   # width of the bar
    baseline_pos_x0 = 2 * epochs                          # start x-position for baseline (bar)
    baseline_pos_x1 = 2 * epochs + basline_width          # end x-position for baseline (bar)
    
    # build triple (iteration, layer, performance)
    D1D1 = list(zip(D1D1_iteration, [layer] * len(D1D1), D1D1))
    D2D2 = list(zip(D2D2_iteration, [layer] * len(D2D2), D2D2))
    mean = list(zip(mean_mode_iteration, [layer] * len(mean_data), mean_data))
    mode = list(zip(mean_mode_iteration, [layer] * len(mode_data), mode_data))
    baseline = list(zip([baseline_pos_x0, baseline_pos_x0 , baseline_pos_x1 , baseline_pos_x1], [layer] * 4, [0, basline_height, basline_height, 0]))
    
    return [D1D1], [D2D2], [mean], [mode], [baseline]

  def plot(self, training='D1D1', retraining=['D2D2', 'D2DAll', 'D2D1'], baseline='baseline'): # 'D2D2' ->'D2D1'
    ''' plot the experiment 
    
    TODO: check dataset identifier (retraining)
    
    @param training: name of dataset (str) to get from experiment for plot in the first part (half) of plot 
    @param retraining: name of the datasets (list(str) with 2 element) to get values from experiment for the second part of the plot
    @return: None, create a pdf file with parameter as name
    '''
    # if 'task_D5-5g_lr_0.001_relr_0.001_h1_800_h2_800_h3_0_ds_Fruits' not in self.params: 
    #  print(self.params)
    #  return
       
    def check_and_load_data(exp, action):
      ''' check data if it seems valid 
      
      TODO: better data check
      '''
      iterations_avg = np.average(exp.data[action][:, 0])
      if iterations_avg <= 0: print('WARNING: average iterations on {} ({} < 0) on {}'.format(action, iterations_avg, exp.params))
      performance_avg = np.average(exp.data[action][:, 1])
      if performance_avg < .1: print('WARNING: average performance on {} ({} < .1) on {}'.format(action, performance_avg, exp.params))
      
      return exp.data[action]
    
    # data integrity check and load action        
    D1D1 = check_and_load_data(self, training)
    D2D2 = check_and_load_data(self, retraining[0])
    D2DAll = check_and_load_data(self, retraining[1])
    D2D1 = check_and_load_data(self, retraining[2])
    baseline_max = check_and_load_data(self, baseline)[:, 1].max() # best baseline performance
    
    plt.figure(figsize=(6, 4))
    axis = plt.gca()
    iterations = D1D1[:, 0].astype(np.float)     # column 0 = iterations for D1D1
    epochs = np.count_nonzero(iterations == 0)   # get number of epochs  (count zero numbers)
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
    
    #---------------------------------------------------------------------------- draw baseline (train D1uD2 test D1uD2)
    y = [baseline_max, baseline_max]    # best value from baseline      
    x = [0, 2 * epochs]                 # from left to right
    axis.plot(x, y, '--k',              # black dashed line
              linewidth=3,
              label='baseline max')
    
    #----------------------------------------------------------------------------------------------------- draw training
    y = D1D1[:, 1].astype(np.float)     # column 1 = performance on D1D1
    x = np.linspace(0.0, epochs, iterations.shape[0], endpoint=False)
    
    # list of marker (every epoch)
    marker = [mark for mark in range(0, iterations.shape[0], last_iter)] 
    marker.append(-1)                            # add line ending
    
    axis.plot(x, y, '^-',               # marker rectangle, with line 
              markevery=marker,      # maker every 10 points
              mfc='white',              # filled white
              markeredgecolor='black',  # edge black
              linewidth=2, color='blue', label='test:$D_1$')
    
    #--------------------------------------------------------------------------------------------- draw re-training D2D2
    iterations = D2D2[:, 0]             # column 0 = iterations for D2D2
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
    # list of marker (every epoch)
    
    y = D2D2[:, 1].astype(np.float)     # column 1 = performance
    x = np.linspace(epochs, 2 * epochs, y.shape[0], endpoint=False)
    
    marker = [mark for mark in range(0, y.shape[0], last_iter)] 
    marker.append(-1)                   # add line ending
    
    axis.plot(x, y, 's-',               # marker circle, with line
              markevery=marker,         # maker every 10 points
              mfc='white',              # filled white
              markeredgecolor='black',  # edge black 
              linewidth=2, color='green', label='test:$D_2$')
    
    #--------------------------------------------------------------------------------------------- draw re-training D2D1
    iterations = D2D1[:, 0]             # column 0 = iterations for D2D1
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
    # list of marker (every epoch)
    
    y = D2D1[:, 1].astype(np.float)     # column 1 = performance
    x = np.linspace(epochs, 2 * epochs, y.shape[0], endpoint=False)
    
    marker = [mark for mark in range(0, y.shape[0], last_iter)] 
    marker.append(-1)                   # add line ending
    
    axis.plot(x, y, '^-',               # marker circle, with line
              markevery=marker,         # maker every 10 points
              markerfacecolor='white',  # filled white
              markeredgecolor='black',  # edge black 
              linewidth=2, color='blue')# no label needed

    #------------------------------------------------------------------------------------------- draw re-training D2DAll
    y = D2DAll[:, 1].astype(np.float)     # column 1 = performance on D2DAll
    # x = np.linspace(epochs, 2 * epochs, y.shape[0], endpoint=False) # same as D2D2
    marker = [mark for mark in range(0, y.shape[0], last_iter)] 
    marker.append(-1)                   # add line ending

    axis.plot(x, y, 'o-',               # marker circle, with line
              # marker=r'$\heartsuit$',
              markevery=marker,         # maker every 10 points
              mfc='white',               # filled white
              markeredgecolor='black',  # edge black
              linewidth=2, color='red', label='test:$D_1\!\cup\!D_2$')
    
    #-------------------------------------------------------------------------------------------------- draw axis labels
    axis.set_xlabel(r'epoch', fontsize=18)    
    axis.set_ylabel(r'test accuracy', fontsize=18)
    axis.tick_params(labelsize=14)
    
    # x-axis labels
    plt.xticks(np.arange(0, 2 * epochs + 1, 2))
    plt.xlim(-.5, 2 * epochs + .5)
    
    # legend position
    axis.legend(fontsize=16, loc='lower left', bbox_to_anchor=(-0.01, -0.01), framealpha=.9)
    
    # enable grid
    axis.grid(True, which='both')
    
    # print gray background for D2
    x = np.arange(epochs, 2 * epochs + 1, 1)
    axis.fill_between(x, 0, 1, where=(x > (x.shape[0] / 2)), facecolor='gray', alpha=0.2)
    
    plt.tight_layout()
    filename = '_'.join(self.fields[:-1])
    plt.savefig('{}.pdf'.format(filename))
    plt.clf()
    plt.close('all')
    
  def plot_light(self, training='D1D1', retraining=['D2D2', 'D2DAll', 'D2D1'], baseline='baseline'): # 'D2D2' ->'D2D1'
    ''' plot the experiment in a detail reduced version for only demonstrating the effect (similar to plot function)
    
    @param training: name of dataset (str) to get from experiment for plot in the first part (half) of plot 
    @param retraining: name of the datasets (list(str) with 3 element) to get values from experiment for the second part of the plot
    @return: None, create a pdf file with parameter as name "<name>_light.pdf"
    '''
    # activate if only one experiment with given name should be plotted 
    # if 'task_D5-5g_lr_0.001_relr_0.001_h1_800_h2_800_h3_0_ds_Fruits' not in self.params: return 
       
    def check_and_load_data(exp, action):
      ''' check data if it seems valid 
      
      TODO: better data check
      '''
      iterations_avg = np.average(exp.data[action][:, 0])
      if iterations_avg <= 0: print('WARNING: average iterations on {} ({} < 0) on {}'.format(action, iterations_avg, exp.params))
      performance_avg = np.average(exp.data[action][:, 1])
      if performance_avg < .1: print('WARNING: average performance on {} ({} < .1) on {}'.format(action, performance_avg, exp.params))
      
      return exp.data[action]
    
    # data integrity check and load action        
    D1D1 = check_and_load_data(self, training)
    D2D2 = check_and_load_data(self, retraining[0])
    D2DAll = check_and_load_data(self, retraining[1])
    D2D1 = check_and_load_data(self, retraining[2])
    
    axis = plt.gca()
    iterations = D1D1[:, 0].astype(np.float)     # column 0 = iterations for D1D1
    epochs = np.count_nonzero(iterations == 0)   # get number of epochs  (count zero numbers)
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
    
    plot_tex_font = False
    marker_size = 15
    
    if plot_tex_font: # use tex font (very slow) (not working)
      from matplotlib import rc
      rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
      rc('text', usetex=True)
    
    #----------------------------------------------------------------------------------------------------- draw training
    y = D1D1[:, 1].astype(np.float)     # column 1 = performance on D1D1
    x = np.linspace(0.0, epochs, iterations.shape[0], endpoint=False)
    
    # list of marker (every epoch)
    marker = [mark for mark in range(0, iterations.shape[0], last_iter * 2)] 
    
    axis.plot(x, y, '^-',               # marker rectangle, with line 
              markevery=marker,         # maker every 10 points
              markerfacecolor='white',  # filled white
              markeredgecolor='black',  # edge black
              markersize=marker_size,
              linewidth=3, color='blue', label='$D_1$')
    
    #--------------------------------------------------------------------------------------------- draw re-training D2D2
    iterations = D2D2[:, 0]             # column 0 = iterations for D2D2
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
    # list of marker (every epoch)
    
    y = D2D2[:, 1].astype(np.float)     # column 1 = performance
    x = np.linspace(epochs, 2 * epochs, y.shape[0], endpoint=False)
    
    marker = [mark for mark in range(0, y.shape[0], last_iter * 2)] 
    marker.append(-1)                   # add line ending
    
    axis.plot(x, y, 's-',               # marker circle, with line
              markevery=marker,         # maker every 10 points
              markerfacecolor='white',  # filled white
              markeredgecolor='black',  # edge black
              markersize=marker_size,
              linewidth=3, color='green', label='$D_2$')
    
    #--------------------------------------------------------------------------------------------- draw re-training D2D1
    iterations = D2D1[:, 0]             # column 0 = iterations for D2D1
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
    # list of marker (every epoch)
    
    y = D2D1[:, 1].astype(np.float)     # column 1 = performance
    x = np.linspace(epochs, 2 * epochs, y.shape[0], endpoint=False)
    
    marker = [mark for mark in range(0, y.shape[0], last_iter * 2)] 
    marker.append(-1)                   # add line ending
    
    axis.plot(x, y, '^-',               # marker circle, with line
              markevery=marker,         # maker every 10 points
              markerfacecolor='white',  # filled white
              markeredgecolor='black',  # edge black 
              markersize=marker_size,
              linewidth=3, color='blue')# no label needed

    #------------------------------------------------------------------------------------------- draw re-training D2DAll
    y = D2DAll[:, 1].astype(np.float)     # column 1 = performance on D2DAll
    # x = np.linspace(epochs, 2 * epochs, y.shape[0], endpoint=False) # same as D2D2 # should be same as D2D1
    marker = [mark for mark in range(0, y.shape[0], last_iter * 2)] 
    marker.append(-1)                   # add line ending

    axis.plot(x, y, 'o-',               # marker circle, with line
              # marker=r'$\heartsuit$', # make love not war
              markevery=marker,         # maker every 10 points
              markerfacecolor='white',  # filled white
              markeredgecolor='black',  # edge black
              markersize=marker_size,
              linewidth=3, color='red', label='$D_1\!\cup\!D_2$')
    
    #-------------------------------------------------------------------------------------------------- draw axis labels
    axis.set_xlabel(r'epoch', fontsize=24)    
    axis.set_ylabel(r'test accuracy', fontsize=24)
    axis.tick_params(labelsize=26, which='major', top=True)
    
    # x-axis labels
    plt.xticks(np.arange(0, 2 * epochs + 1, 5), ('$0$', '', r'$\mathcal{E}$', '', r'$2\mathcal{E}$'), ha='center')
    plt.xlim(-.5, 2 * epochs + .5)
    
    # clone axis for top (only naming of white (traing D1) and gray (retraining D2))
    axis2 = axis.twiny()                                     # make copy 
    axis2.tick_params(labelsize=20, which='major', top=True) # only top 
    axis2.set_xlim(axis.get_xlim())                          # copy limits
    axis2.set_xticks(np.arange(0, 2 * epochs + 1, 5))        # set points
    axis2.set_xticklabels(['', r'$\mathit{training\ }D_1$', '', r'$\mathit{retraining\ }D_2$', '']) # rename labels
    
    # y-axis
    plt.yticks(np.arange(.0, 1.0 + 0.5, .5))
    
    # legend
    axis.legend(fontsize=22,
                labelspacing=.2, # reduce line spacing  
                loc='lower left',
                bbox_to_anchor=(0.02, 0.02), framealpha=.9)
    
    # enable grid
    axis.grid(True, which='both')
    
    # print gray background for retraining
    x = np.arange(epochs, 2 * epochs + 1, 1)
    axis.fill_between(x, 0, 1, where=(x > (x.shape[0] / 2)), facecolor='gray', alpha=0.2)
    
    plt.tight_layout(pad=2)
    filename = '_'.join(self.fields[:-1])
    plt.savefig('{}_light.pdf'.format(filename))
    plt.clf()
  
  def create_retrain_experiments(self):
    ''' call doExperiments with own parameters to recreate new experiments for second run (fix parameters, all retrain learning rates)
    
    existing files get appended
    '''
    # print('CALL: python E:\workspace\python\IncrementalLearning\doExperiments.py --fix_params {}'.format('_'.join(self.fields)))
    os.system('python E:\workspace\python\IncrementalLearning\doExperiments.py --fix_params {}'.format('_'.join(self.fields)))
    
  def recreate_experiment(self):
    ''' call doExperiments with own parameters to recreate experiment
    existing files get appended
    
    @deprecated: should be removed, only for bugfix in dnn.py
    '''
    print('CALL: python E:\workspace\python\IncrementalLearning\doExperiments.py --fix_params {}'.format('_'.join(self.fields)))
    os.system('python E:\workspace\python\IncrementalLearning\doExperiments.py --recreate {}'.format('_'.join(self.fields)))
  
  @classmethod
  def get_exp_lookup_table(cls):
    return cls._results[cls.MATRIX_LOOKUP_TABLE]

  @classmethod
  def matrix_lookup(cls, task_id, param_id, dataset_id):
    ''' return the (previous calculated and stored in matrix lookup table) best performance 
     
    @param task_id: (int)
    @param param_id: (int) 
    @param dataset_id: (int)
    '''
    return cls._results[cls.MATRIX_LOOKUP_TABLE][task_id, param_id, dataset_id]

  def _set_result(self, result_matrix_id, result_value):
    ''' store the result value in the corresponding result matrix
    
    @param result_matrix_id: key (str) for the target matrix in Experiments._results e.g. 'resultMatrixTrain' (measure function name)
    @param result_value: result value (float) 
    '''
    self._results[result_matrix_id] = result_value
    Experiment._results[result_matrix_id][self.__task_id, self.__param_id, self.__dataset_id] = result_value

  def load_result_value(self, result_matrix_id):
    return Experiment._results[result_matrix_id][self.__task_id, self.__param_id, self.__dataset_id]

  @classmethod
  def make_fix_parameter_string(cls, params):
    ''' substitute in params string the fix parameters with "<param_name>_*_"
    e.g.: 
            lr_0.001_relr_0.001_h1_200_h2_800_h3_400_ds_MNIST -> subst [ "relr", "h2" ] 
            lr_0.001_relr_*_h1_200_h2_*_h3_400_ds_MNIST
            
            Experiment.fix_parameter_exp = {'lr_0.001_relr_*_h1_200_h2_*_h3_400_ds_MNIST' : self}
    '''
    for fix_parameter, fix_parameter_pattern in Experiment.fix_parameters_pattern.items():
      params = re.sub(fix_parameter_pattern, fix_parameter + '_*_', params)
        
    return params

  @classmethod    
  def resolve_fix_parameter_string(cls, fix_params):
    if '*' not in fix_params: 
      fix_params = cls.make_fix_parameter_string(fix_params)
    print('search for experiments with fixed_params:', fix_params)
    return cls.fix_parameter_exp[fix_params]
      
  @classmethod
  def __precompile_pattern(cls, fix_parameters):
    ''' create precompiled regex pattern for substitution of parameters 
    
    e.g. lr_0.001_relr_0.0001_h1_800_h2_200_h3_800_ds_MNIST (subst "relr") 
         
    '''
    for fix_parameter in fix_parameters:
      pattern = re.compile(r'{}_.*?_'.format(fix_parameter))
      cls.fix_parameters_pattern[fix_parameter] = pattern
  
  @classmethod
  def __build_lookup_tables(cls):
    ''' create dictionaries that map tasks (e.g. D5-5b) to integers
        create dictionaries that map parameter sets (e.g. lr_0.001_relr_0.001_h1_200_h2_200_h3_0_ds_notMNIST) to integers
        and vice versa (inv_*)

    task_lookup_table:             {task (str) : task_id (int)}
    task_lookup_table e.g.:        {'D9-1b': 6, 'DP10-10': 5, 'D5-5d': 8, ...}
     
    inv_task_lookup_table:         {task_id (int): task (str)}
    inv_task_lookup_table e.g.:    {0: 'D5-5b', 1: 'DP5-5', 2: 'DP10-10', ...}
    
    
    param_lookup_table:            {params (str): param_id (int)}
    param_lookup_table e.g.:       {'0.0001_retrainlr_0.001_layers_200_800_0_CIFAR-10': 270, '0.0001_retrainlr_0.001_layers_800_800_0_fashionMNIST': 367, ...}
    
    inv_param_lookup_table:        {param_id (int): params (str)}
    inv_param_lookup_table e.g.:   {0: '1e-05_retrainlr_0.001_layers_200_200_400_CIFAR-10', 1: '1e-05_retrainlr_0.001_layers_200_200_800_notMNIST', ...}
    
    dataset_lookup_table:          {dataset (str): dataset_id (int)}
    dataset_lookup_table e.g.:     {'notMNIST': 0, 'MNIST': 1, 'CIFAR-10': 2, 'fashionMNIST': 3}
    
    inv_dataset_lookup_table:      {dataset_id (int): dataset (str)}
    inv_dataset_lookup_table e.g.: {0: 'notMNIST', 1: 'MNIST', 2: 'CIFAR-10', 3: 'fashionMNIST'}
    
    '''
    # build task_lookup_table: { enumerate(task_lookup_table) } (convert set to dict)
    cls.task_lookup_table = { task:task_id for task_id, task in enumerate(cls.task_lookup_table) }
    # invert task_lookup_table 
    cls.inv_task_lookup_table = { task:task_id for task_id, task in cls.task_lookup_table.items() }
    
    # build param_lookup_table: { enumerate(param_lookup_table) } (convert set to dict)
    cls.param_lookup_table = { param:param_id for param_id, param in enumerate(cls.param_lookup_table) }
    # invert param_lookup_table
    cls.inv_param_lookup_table = { param:param_id for param_id, param in cls.param_lookup_table.items() }
    
    # build dataset_lookup_table: { enumerate(dataset_lookup_table) } (convert set to dict)
    cls.dataset_lookup_table = { dataset:dataset_id for dataset_id, dataset in enumerate(cls.dataset_lookup_table) }
    # invert dataset_lookup_table
    cls.inv_dataset_lookup_table = { dataset_id:dataset for dataset, dataset_id in cls.dataset_lookup_table.items() }
  
  @classmethod
  def build_experiments(cls, path, model=None, dataset_name=None, task=None, fix_parameter=['relr']):
    ''' create for all experiments a new Experiment-object with all related files
    
    1. search for all csv-files in given path
        1.1. if needed, filter only one experiment
        1.2. if needed, filter only one dataset
    2. create exclusive one Experiment-object for the first file
    3. add related files to existing objects
    4. add dataset type to a set
    5. create lookup tables  
    
    e.g.: 
        filename = LWTA-fc-MRL_task_D5-5_lr_0.001_relr_0.001_h1_200_h2_400_h3_200_ds_MNIST_D2D1.csv
                  ^-----------------------------------------------------------------------^----^
                                         run_id (could grow)                              action      
    
    @param path: path to search for files (str) 
    @param model: identifier (str) of experiment e.g. EWC or fc, default None: all experiments will be processed
    @param dataset_name: identifier (str) of dataset e.g. MNIST, default None: all dataset will be processed
    @param task: identifier (str) to select a single task, None: all tasks will be processed
    @param fix_parameter: this parameter(s) [str] get optimized 
    '''
    cls.__precompile_pattern(fix_parameter)
    
    # add os.sep if not exists
    if not path.endswith(os.sep): path = path + os.sep
    
    # iterate all files in path
    exp_id = 0
    all_files = os.listdir(path)
    
    print('create Experiments..', end='', flush=True)
    for filename in all_files:
      # skip files without suffix ".csv"
      if not filename.endswith('.csv'): continue 
      
      # filter model if not None
      if model is not None and not filename.startswith(model): continue
      if task is not None and not task in filename: continue
      
      filename_without_extension = filename.replace('.csv', '')
      fields = filename_without_extension.split('_')
      dataset = fields[-2]
      run_id = '_'.join(fields[:-1])
      
      # filter dataset if not None
      if dataset_name is not None and dataset_name != dataset: continue
      
      # create Experiment-objects or add files to an existing
      if run_id not in Experiment.expDict:
        cls.expDict[run_id] = Experiment(run_id, exp_id, path + filename)
        exp_id += 1
      else:
        cls.expDict[run_id]._add_file(path + filename)
       
      cls.dataset_lookup_table.add(dataset) # create list of datasets for dataset lookup table
    
    print('done ({})..'.format(len(cls.expDict)), end='', flush=True)
    
    # build lookup tables
    print(' build lookup tables..', end='', flush=True) 
    cls.__build_lookup_tables()
    
    # read all files
    print(' read files..', end='', flush=True)
    for experiment in cls.expDict.values(): experiment._read_files() 
    
    # do self check
    print(' selfcheck (failed: ', end='', flush=True)
    for experiment in cls.expDict.values(): experiment._self_check() 
    print('{})'.format(len(cls.__fails)), end='', flush=True)
    
    # create lookup matrix, load mapping and add to lookup matrix
    print(' create matrix lookup table.. ', end='', flush=True)
    cls.create_matrix(cls.MATRIX_LOOKUP_TABLE, np.object)
    for experiment in cls.expDict.values(): experiment._load_mapping() 
      
  @classmethod
  def create_matrix(cls, key, dtype=np.float64):
    num_tasks = len(cls.task_lookup_table)
    num_params = len(cls.param_lookup_table)
    num_datasets = len(cls.dataset_lookup_table)
    
    result_matrix = np.full((num_tasks, num_params, num_datasets), -1., dtype=dtype)
    
    cls._results[key] = result_matrix

  @classmethod
  def get_result_matrix(cls, key):
    return cls._results[key]
  
  @classmethod
  def print_result_matrix(cls, key):
    np.set_printoptions(threshold=np.nan)
    print(cls._results[key])
  
