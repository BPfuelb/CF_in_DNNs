'''
Created on 01.06.2018

@author: BPF
'''
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from colour import Color
from matplotlib import colors as mcolors
import os

from eval.experiment import Experiment as Exp 
import matplotlib.pyplot as plt
import numpy as np


class Experiment_Plotter():
  ''' provide function to plot single experiments (only IMM, other were ploted by Experiment class self) or 
    multiple experiments for fix task but different datasets '''

  @classmethod
  def plot_imm_exp(cls, experiment, _mode, training='D1D1', retraining=['D2D2', 'D2DAll'], baseline='baseline'):
    ''' plot the experiment (comparable with plot function in experiment.py)
        
    TODO: check dataset identifier (retraining)
    
    @param training: name of dataset (str) to get from experiment for plot in the first part (half) of plot 
    @param retraining: name of the datasets (list(str) with 2 element) to get values from experiment for the second part of the plot
    @return: None, create a pdf file with parameter as name
    '''
    
    def check_and_load_data(exp, action):
      ''' check data if it seems valid (only rough) '''
      iterations_avg = np.average(exp.data[action][:, 0])
      if iterations_avg <= 0: print('WARNING: average iterations on {} ({} < 0) on {}'.format(action, iterations_avg, exp.params))
      performance_avg = np.average(exp.data[action][:, 1])
      if performance_avg < .1: print('WARNING: average performance on {} ({} < .1) on {}'.format(action, performance_avg, exp.params))
      
      return exp.data[action]
    
    # data integrity check and load action        
    D1D1 = check_and_load_data(experiment, training)
    D2D2 = check_and_load_data(experiment, retraining[0])
    D2D1 = check_and_load_data(experiment, retraining[1])
    baseline_max = check_and_load_data(experiment, baseline)[:, 1].max() # best baseline performance
    
    # Create two subplots sharing y axis
    _, (training, alpha) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios':[1, 1]}) 
    
    iterations = D1D1[:, 0].astype(np.float)     # column 0 = iterations for D1D1
    epochs = np.count_nonzero(iterations == 0)   # get number of epochs  (count zero numbers)
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
      
    #---------------------------------------------------------------------------- draw baseline (train D1uD2 test D1uD2)
    y = [baseline_max, baseline_max]       # best value from baseline      
    x = [0, 2 * epochs]                    # from left to right
    training.plot(x, y, '--k',             # black dashed line
                  linewidth=3,
                  label='baseline best')   
                                           
    #----------------------------------------------------------------------------------------------------- draw training
    y = D1D1[:, 1].astype(np.float)        # column 1 = performance on D1D1
    x = np.linspace(0.0, epochs, iterations.shape[0], endpoint=False)
    
    # list of marker (every epoch)
    marker = [mark for mark in range(0, iterations.shape[0], last_iter)] 
    marker.append(-1)                      # add last marker
                                          
    training.plot(x, y, '^-',              # marker rectangle, with line 
                  markevery=marker,        # maker every 10 points
                  mfc='white',             # filled white
                  markeredgecolor='black', # edge black
                  linewidth=2, color='blue', label='train:D1;test:D1')
    
    #--------------------------------------------------------------------------------------------- draw re-training D2D2
    iterations = D2D2[:, 0]                 # column 0 = iterations for D2D2
    last_iter = int((iterations.max() + iterations[1]) / iterations[1])
    # list of marker (every epoch)
    
    y = D2D2[:, 1].astype(np.float)         # column 1 = performance
    x = np.linspace(epochs, 2 * epochs, y.shape[0], endpoint=False)
    
    marker = [mark for mark in range(0, y.shape[0], last_iter)] 
    marker.append(-1)                       # add last marker
                                            
    training.plot(x, y, 's-',               # marker circle, with line
                  markevery=marker,         # maker every 10 points
                  mfc='white',              # filled white
                  markeredgecolor='black',  # edge black 
                  linewidth=2, color='green', label='train:D2;test:D2')
    
    #------------------------------------------------------------------------------------------- draw re-training D2DAll
    mean = D2D1[::2]                        # all mean imm [alpha,performance] values
    mode = D2D1[1::2]                       # all mode imm [alpha,performance] values
                                           
    y_mean = mean[:, 1]                     # only performance values for mean
    y_mode = mode[:, 1]                     # only performance values for mean
                                          
    x_mean = mean[:, 0]                     # only alpha values for mean                     
    x_mode = mode[:, 0]                     # only alpha values for mode
    
    # marker for mean and mode performance
    marker = [mark for mark in range(0, y_mean.shape[0], last_iter)]
    marker.append(-1)                       # add last marker 
    
    # best mean, best mode
    best_mean = mean[y_mean.argmax()]
    best_mode = mode[y_mode.argmax()]
    
    #------------------------------------------------------------------------------------------------ performance values
    # plot mean values
    alpha.plot(x_mean, y_mean, 'o-',        # marker circle, with line
               markevery=marker,            # maker every 10 points
               mfc='white',                 # filled white
               markeredgecolor='black',     # edge black
               linewidth=1, color='red', label='Mean-IMM;test:All')
    
    # plot mode values
    alpha.plot(x_mode, y_mode, 'v-',        # marker rectangle, with line
               markevery=marker,            # maker every 10 points
               mfc='white',                 # filled white
               markeredgecolor='black',     # edge black
               linewidth=1, color='orange', label='Mode-IMM;test:All')
    
    #------------------------------------------------------------------------------------------------------- annotations
    
    # upper annotation for higher (max) value, lower annotation for lower (max) value 
    if best_mean[1] > best_mode[1]: # TODO: dynamic position with 'textcoords' or 'horizontalalignment'
      text_mode_pos = (.7, .05)
      text_mean_pos = (0, .95)
    else:
      text_mode_pos = (0, 0.95)
      text_mean_pos = (.7, .05)
      
    # annotate mean maximum (good annotation guide: https://matplotlib.org/users/annotations.html)
    alpha.annotate('Mean $\\alpha$={:.2f}, max={:.2f}'.format(*best_mean), # text 
                   xy=(best_mean[0], best_mean[1]),                    # target value position
                   xytext=text_mean_pos,                               # text position
                   bbox=dict(boxstyle="round", fc="w"),                # white box for text
                   arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90,rad=5"), # line style
                   fontsize=8)
    
    # annotate mode maximum
    alpha.annotate('Mode $\\alpha$={:.2f}, max={:.2f}'.format(*best_mode), # text                  
                   xy=(best_mode[0], best_mode[1]),                    # target value position 
                   xytext=text_mode_pos,                               # text position         
                   bbox=dict(boxstyle="round", fc="w"),                # white box for text    
                   arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90,rad=5"), # line style
                   fontsize=8)
    
    #-------------------------------------------------------------------------------------------------- draw axis labels
    training.set_xlabel('epoch', size=22)    
    training.set_ylabel('test accuracy', size=22)
    training.tick_params(labelsize=10)
    training.set_xticks(np.arange(0, 2 * epochs + 1, 2))
        
    alpha.set_xlabel('alpha', size=22)    
    alpha.set_ylabel('test accuracy', size=22)
    alpha.tick_params(labelsize=10)
    
    # x/y-axis limits
    training.set_xlim(-.25, 2 * epochs + .25)
    training.set_ylim(-.025, 1.025)
    
    alpha.set_xlim(-0.025, 1.025)
    alpha.set_ylim(-.025, 1.025)
    
    # create legend
    training.legend(fontsize=12, loc='lower left', bbox_to_anchor=(-0.01, -0.01), framealpha=1.)
    alpha.legend(fontsize=12, loc='lower left', bbox_to_anchor=(-0.01, -0.01), framealpha=1.)
    
    # enable grid
    training.grid(True, which='both')
    alpha.grid(True, which='both')
    
    # print gray background for "retraining"
    x = np.arange(epochs, 2 * epochs + 1, 1)
    training.fill_between(x, 0, 1, where=(x > (x.shape[0] / 2)), facecolor='gray', alpha=0.2)
    
    plt.tight_layout()
    filename = '_'.join(experiment.fields[:-1])
    plt.savefig('{}_{}.pdf'.format(filename, _mode))
    plt.clf()
    plt.close()

  @classmethod
  def plot_all_exp_imm(cls, experiments, _mode):
    ''' plot function for imm model, additionally plot the accuracy for logged alpha values for mean and mode '''
    num_exps_per_task = len(next(iter(experiments.values()))) # all should have the same size (number of experiments)
    
    def generate_colors(color_start, color_end, alpha_start=0.9, alpha_end=1.0, edge_color='w'):
      ''' generate different colors for layers (faces and edges) ''' 
      light = Color(color_start)  
      dark = Color(color_end) 
      colors = list(light.range_to(dark, num_exps_per_task))
      alpha = np.arange(alpha_start, alpha_end, (alpha_end - alpha_start) / num_exps_per_task)
      alpha[-1] = 1.0
      
      face = [ mcolors.to_rgba(color.__str__(), alpha=alpha[layer]) for layer, color in enumerate(colors) ]
      edge = [ mcolors.to_rgba(edge_color, alpha=1) for _ in range(num_exps_per_task) ]
      return face, edge
  
    face_D1D1, edge_D1D1 = generate_colors('#004A7F', '#004A7F', 0.5, 0.8)                     # blue
    face_D2D2, edge_D2D2 = generate_colors('#38920C', '#38920C', 0.5, 0.8)                     # green
    face_mode, edge_mode = generate_colors('#ed0000', '#930202', 0.9)                          # red
    face_mean, edge_mean = generate_colors('#f4a742', '#f48a00', 0.5)                          # orange    
    face_baseline, edge_baseline = generate_colors('#F0F0F0', '#FFFFFF', 0.9, edge_color='k')  # white
    
    for _, all_exp_task in experiments.items(): # for all different tasks create one image 
      fig = plt.figure(figsize=(16, 9))
      ax = fig.add_subplot(111, projection='3d')
      
      all_exp_task = sorted(all_exp_task, key=lambda x: np.average(x.data['D2DAll'][:, 1]))
      for layer, exp in enumerate(all_exp_task):
        D1D1, D2D2, mean, mode, baseline = exp.create_plot_vertices_imm(layer) # get data from experiment
        
        poly_D1D1 = Poly3DCollection(D1D1, facecolors=face_D1D1[layer], edgecolors=edge_D1D1[layer], linewidths=1.0)
        poly_D2D2 = Poly3DCollection(D2D2, facecolors=face_D2D2[layer], edgecolors=edge_D2D2[layer], linewidths=1.0)
        poly_mean = Poly3DCollection(mean, facecolors=face_mean[layer], edgecolors=edge_mean[layer], linewidths=1.0)
        poly_mode = Poly3DCollection(mode, facecolors=face_mode[layer], edgecolors=edge_mode[layer], linewidths=1.0)
        poly_baseline = Poly3DCollection(baseline, facecolors=face_baseline[layer], edgecolors=edge_baseline[layer], linewidths=1.0)
      
        poly_D1D1.set_sort_zpos(num_exps_per_task - 2 * layer)
        poly_D2D2.set_sort_zpos(num_exps_per_task - 2 * layer)
        poly_mean.set_sort_zpos(num_exps_per_task - 2 * layer + 1)
        poly_mode.set_sort_zpos(num_exps_per_task - 2 * layer + 0.9)
        poly_baseline.set_sort_zpos(num_exps_per_task - 2 * layer + 1.1)
      
        ax.add_collection3d(poly_D2D2, zdir='y')
        ax.add_collection3d(poly_mean, zdir='y')
        ax.add_collection3d(poly_mode, zdir='y')
        ax.add_collection3d(poly_D1D1, zdir='y')
        ax.add_collection3d(poly_baseline, zdir='y')

      #---------------------------------------------------------- configure axes
      # x axis
      ax.set_xlabel('\n\n         epoch | alpha')
      ax.set_xlim3d(0, 40 + 1)
      x_ticks = np.arange(0, 40 + 3, 2)
      
      epoch_list = [str(x) for x in range(0, 21, 2)]
      alpha_list = ['{:.1f}'.format(x) if x % 0.2 < 0.05 or x == 1.0 else '' for x in np.arange(0, 1.1, 0.1) ]
      ax.set_xlim3d(0, 40 + 2)
      ax.set_xticks(x_ticks)
      ax.set_xticklabels(epoch_list + alpha_list)
      
      # y axis
      ax.set_ylabel('\ndataset', linespacing=5.0)
      ax.set_ylim3d(0, num_exps_per_task)
      ax.set_yticks([ i for i in range(num_exps_per_task) ])
      ax.set_yticklabels(['\n\n' + (' ' * 6) + '{:<14}'.format(exp.dataset if exp.dataset != 'Devnagari' else 'Devanagari') for exp in all_exp_task], linespacing=-20.4, rotation=-10) # not working
      ax.yaxis._axinfo['label']['space_factor'] = 3.0
      
      # z axis
      ax.set_zlim3d(0.0, 1.0)
      ax.set_zlabel('\naccuracy', linespacing=-6.0)
      ax.set_zticks([0.0 , 0.5, 1.0])
      ax.set_zticklabels(['\n{:<9}'.format(performance) for performance in [0, 0.5, 1.0]], linespacing=-.4)
      
      ax.view_init(30, -45)        # elevation, rotation (-90 -> frontal)
          
      plt.subplots_adjust(top=0.4) # vertical compressing
      
      # create legend
      blue = plt.Rectangle((0, 0), 1, 1, fc=face_D1D1[-1], edgecolor='k')
      green = plt.Rectangle((0, 0), 1, 1, fc=face_D2D2[-1], edgecolor='k')
      red = plt.Rectangle((0, 0), 1, 1, fc=face_mean[-1], edgecolor='k')
      orange = plt.Rectangle((0, 0), 1, 1, fc=face_mode[-1], edgecolor='k')
      white = plt.Rectangle((0, 0), 1, 1, fc=face_baseline[-1], edgecolor='k')
      ax.legend([blue, green, red, orange, white], ['$D_1$', '$D_2$', 'mean', 'mode', 'max. baseline'],
                bbox_to_anchor=(0.18, 0.2))
      
      first_exp = all_exp_task[0]
      plt.savefig('{}_{}_{}.pdf'.format(first_exp.fields[0], first_exp.task, _mode), format='pdf', bbox_inches='tight')
      plt.clf()
      plt.close('all')

  @classmethod
  def plot_all_exp(cls, experiments):
    ''' 3d plot for similar experiments for different datasets (no IMM) '''
    # num_tasks = len(experiments)
    num_exps_per_task = len(next(iter(experiments.values()))) # all should have the same size (number of experiments)
    
    def generate_colors(color_start, color_end, alpha_start=0.9, alpha_end=1.0, edge_color='w'):
      ''' generate different colors for layers (faces and edges) ''' 
      light = Color(color_start)  
      dark = Color(color_end) 
      colors = list(light.range_to(dark, num_exps_per_task))
      alpha = np.arange(alpha_start, alpha_end, (alpha_end - alpha_start) / num_exps_per_task)
      alpha[-1] = 1.0
      
      face = [ mcolors.to_rgba(color.__str__(), alpha=alpha[layer]) for layer, color in enumerate(colors) ]
      edge = [ mcolors.to_rgba(edge_color, alpha=1) for _ in range(num_exps_per_task) ]
      return face, edge
  
    face_D1D1, edge_D1D1 = generate_colors('#004A7F', '#004A7F', 0.5, 0.8)                     # blue
    face_D2D2, edge_D2D2 = generate_colors('#38920C', '#38920C', 0.5, 0.8)                     # green
    face_D2DAll, edge_D2DAll = generate_colors('#ed0000', '#930202', 0.9)                      # red  
    face_baseline, edge_baseline = generate_colors('#F0F0F0', '#FFFFFF', 0.9, edge_color='k')  # white
    
    for _, all_exp_task in experiments.items(): # for all different tasks create one image 
      fig = plt.figure(figsize=(16, 9))
      ax = fig.add_subplot(111, projection='3d')
      
      all_exp_task = sorted(all_exp_task, key=lambda x: np.max(x.data['baseline'][:, 1]))
      for layer, exp in enumerate(all_exp_task):
        D1D1, D2D2, D2DAll, baseline = exp.create_plot_vertices(layer) # get data from experiment
        
        polyD1D1 = Poly3DCollection(D1D1, facecolors=face_D1D1[layer], edgecolors=edge_D1D1[layer], linewidths=1.0)
        polyD2D2 = Poly3DCollection(D2D2, facecolors=face_D2D2[layer], edgecolors=edge_D2D2[layer], linewidths=1.0)
        polyD2DAll = Poly3DCollection(D2DAll, facecolors=face_D2DAll[layer], edgecolors=edge_D2DAll[layer], linewidths=1.0)
        polyBaseline = Poly3DCollection(baseline, facecolors=face_baseline[layer], edgecolors=edge_baseline[layer], linewidths=1.0)
      
        polyD1D1.set_sort_zpos(num_exps_per_task - 2 * layer)
        polyD2D2.set_sort_zpos(num_exps_per_task - 2 * layer)
        polyD2DAll.set_sort_zpos(num_exps_per_task - 2 * layer + 1)
        polyBaseline.set_sort_zpos(num_exps_per_task - 2 * layer + 1.1)
      
        ax.add_collection3d(polyD2D2, zdir='y')
        ax.add_collection3d(polyD2DAll, zdir='y')
        ax.add_collection3d(polyD1D1, zdir='y')
        ax.add_collection3d(polyBaseline, zdir='y')

      #---------------------------------------------------------- configure axes
      # x axis
      ax.set_xlabel('\n\nepoch')
      ax.set_xlim3d(0, 20 + 1)
      ax.set_xticks(np.arange(0, 20 + 1, 2))
      ax.yaxis._axinfo['label']['space_factor'] = 3.0
      
      # y axis
      ax.set_ylabel('\ndataset', linespacing=5.0)
      ax.set_ylim3d(0, num_exps_per_task)
      ax.set_yticks([ i for i in range(num_exps_per_task) ])
      # TODO: workaround for typo
      ax.set_yticklabels(['\n\n' + (' ' * 6) + '{:<14}'.format(exp.dataset if exp.dataset != 'Devnagari' else 'Devanagari') for exp in all_exp_task], linespacing=-20.4, rotation=-10) # not working
      
      # z axis
      ax.set_zlim3d(0.0, 1.0)
      ax.set_zlabel('\naccuracy', linespacing=-6.0)
      ax.set_zticks([0.0 , 0.5, 1.0])
      ax.set_zticklabels(['\n{:<9}'.format(performance) for performance in [0, 0.5, 1.0]], linespacing=-.4)
      
      ax.view_init(30, -45)        # elevation, rotation (-90 -> frontal)
          
      plt.subplots_adjust(top=0.4) # vertical compressing
      
      # create legend
      blue = plt.Rectangle((0, 0), 1, 1, fc=face_D1D1[-1], edgecolor='k')
      green = plt.Rectangle((0, 0), 1, 1, fc=face_D2D2[-1], edgecolor='k')
      red = plt.Rectangle((0, 0), 1, 1, fc=face_D2DAll[-1], edgecolor='k')
      white = plt.Rectangle((0, 0), 1, 1, fc=face_baseline[-1], edgecolor='k')
      ax.legend([blue, green, red, white], ['$D_1$', '$D_2$', '$D_1\!\cup D_2$', 'max. baseline'], bbox_to_anchor=(0.18, 0.2))
      
      first_exp = all_exp_task[0]
      plt.savefig('{}_{}.pdf'.format(first_exp.fields[0], first_exp.task), format='pdf', bbox_inches='tight')
      plt.clf()

  @classmethod
  def create_LaTeX_table(cls, experiments):
    ''' create a LaTeX table column from: 
        * min(max(strip_non_numeric(task))
        * min(last(strip_non_numeric(task))
                
      1. all experiments sort by dataset   
      2. group by strip_non_numeric(task) D5-5a, ..., D5-5h -> D5-5
      3. find experiment with max. performance on D1 U D2
      4. extract minimum value on D1 U D2 and last 
    '''
    print()
    # 1. all experiments sort by dataset
    by_dataset = {} # sorted {'dataset_name': [exp] }
    
    for all_exps in experiments.values():
      for exp in all_exps:
        if exp.dataset not in by_dataset: by_dataset[exp.dataset] = []
        by_dataset[exp.dataset].append(exp)
        
    # 2. group by strip_non_numeric(SLT) D5-5a, ..., D5-5h -> D5-5
    by_dataset_and_task = {} # sorted {'dataset_name': {'D5-5': [exp], 'D9-1', [exp], 'DP10-10':[exp]}}
    for dataset, all_exps in by_dataset.items():
      if dataset not in by_dataset_and_task: by_dataset_and_task[dataset] = {}
      
      for exp in all_exps:
        # strip task name
        task_name = exp.task
        if not task_name[-1].isdigit(): # last char is number
          task_name = task_name[:-1]

        # group experiments by striped task name
        if task_name not in by_dataset_and_task[dataset]: by_dataset_and_task[dataset][task_name] = []
        by_dataset_and_task[dataset][task_name].append(exp)
    
    # 3. & 4. find experiment with max. performance on D1 U D2 for each task
    # the max. performance for the last value ([-1])  
    best_dataset_and_task = {}
    datasets = sorted(by_dataset_and_task.keys())
    
    for dataset in datasets:
      tasks = by_dataset_and_task[dataset]
       
      if dataset not in best_dataset_and_task: best_dataset_and_task[dataset] = {}
      
      tasks_keys = sorted(tasks.keys()) 
      for task_key in tasks_keys:
        experiments = by_dataset_and_task[dataset][task_key] # list of all experiments for a aggregated task
        min_exp = sorted(experiments, key=lambda x: np.max(x.data['D2DAll'][:, 1]))[0] # [0]=min [-1]=max
        min_ = min_exp.data['D2DAll'][:, 1].max()
        
        last_exp = sorted(experiments, key=lambda x: x.data['D2DAll'][:, 1][-1])[0] # [0]=min [-1]=max
        last_ = last_exp.data['D2DAll'][:, 1][-1]
        print('{:<13}:{:<8} & {:.2f}/{:.2f}\n'.format(dataset, task_key, min_, last_), end='', flush=True)
    
  @classmethod
  def create_LaTeX_table_imm(cls, experiments, _mode):
    ''' create a LaTeX table column for IMM from (skip DP10-10): 
        * min(max(strip_non_numeric(task))
        * min(last(strip_non_numeric(task))
                
      1. all experiments sort by dataset   
      2. group by strip_non_numeric(task) D5-5a, ..., D5-5h -> D5-5
      3. find experiment with max. performance on D1 U D2
      4. extract minimum value on D1 U D2 and last 
    '''
    print('{} evaluation\n'.format(_mode))
    # 1. all experiments sort by dataset
    by_dataset = {} # sorted {'dataset_name': [exp] }
    
    for all_exps in experiments.values():
      for exp in all_exps:
        if exp.dataset not in by_dataset: by_dataset[exp.dataset] = []
        by_dataset[exp.dataset].append(exp)
        
    # 2. group by strip_non_numeric(SLT) D5-5a, ..., D5-5h -> D5-5
    by_dataset_and_task = {} # sorted {'dataset_name': {'D5-5': [exp], 'D9-1', [exp], 'DP10-10':[exp]}}
    for dataset, all_exps in by_dataset.items():
      if dataset not in by_dataset_and_task: by_dataset_and_task[dataset] = {}
      
      for exp in all_exps:
        # strip task name
        task_name = exp.task
        if not task_name[-1].isdigit(): # last char is number
          task_name = task_name[:-1]

        # group experiments by striped task name
        if task_name not in by_dataset_and_task[dataset]: by_dataset_and_task[dataset][task_name] = []
        by_dataset_and_task[dataset][task_name].append(exp)
    
    # 3. & 4. find experiment with max. performance on D1 U D2 for each task
    # the max. performance for the last value ([-1])  

    best_dataset_and_task = {}
    datasets = sorted(by_dataset_and_task.keys())
    print('mode', end='', flush=True)
    for dataset in datasets:
      if 'Fruits' in dataset: continue
      tasks = by_dataset_and_task[dataset]
      if dataset not in best_dataset_and_task: best_dataset_and_task[dataset] = {}
      tasks_keys = sorted(tasks.keys())
      
      for task_key in tasks_keys: # mode 
        if 'DP10-10' in task_key: continue
        
        experiments = by_dataset_and_task[dataset][task_key] # list of all experiments for a aggregated task
        
        mode_min_exp = sorted(experiments, key=lambda x: np.max(x.data['D2DAll'][1::2][:, 1]))[0] # mode [0]=min [-1]=max
        mode = mode_min_exp.data['D2DAll'][1::2]
        y_mode = mode[:, 1]
        best_mode = mode[y_mode.argmax()][1]
        print(' & {:.2f} '.format(best_mode), end='', flush=True)
    
    best_dataset_and_task = {}
    datasets = sorted(by_dataset_and_task.keys())
    print('\nmean', end='', flush=True) 
    for dataset in datasets:
      if 'Fruits' in dataset: continue
      tasks = by_dataset_and_task[dataset]
      if dataset not in best_dataset_and_task: best_dataset_and_task[dataset] = {}
      tasks_keys = sorted(tasks.keys())
      
      for task_key in tasks_keys:
        if 'DP10-10' in task_key: continue
        experiments = by_dataset_and_task[dataset][task_key] # list of all experiments for a aggregated task

        mean_min_exp = sorted(experiments, key=lambda x: np.max(x.data['D2DAll'][::2][:, 1]))[0] # mean [0]=min [-1]=max 
        mean = mean_min_exp.data['D2DAll'][::2]
        y_mean = mean[:, 1]
        best_mean = mean[y_mean.argmax()][1]
        print(' & {:.2f} '.format(best_mean), end='', flush=True)
      
    print()

  @classmethod
  def load_one_exp(cls, path, exp_name):
    ''' only load one experiment and plot it (only for testing)
    e.g.: 
        filename = LWTA-fc-MRL_task_D5-5_lr_0.001_relr_0.001_h1_200_h2_400_h3_200_ds_MNIST_D2D1.csv
                  ^-----------------------------------------------------------------------^----^
                                         run_id (could grow)                              action      
    
    @deprecated: only for testing (will be removed in further version)
    @param path: path to search for files (str) 
    @param model: identifier (str) of experiment e.g. EWC or fc, default None: all experiments will processed
    @param dataset: identifier (str) of dataset e.g. MNIST, default None: all dataset will processed
    '''
    # TODO: add assertions e.g. test path exists  
    
    # add os.sep if not exists
    if not path.endswith(os.sep): path = path + os.sep
    
    # iterate all files in path
    exp_id = 0
    all_files = os.listdir(path)
    
    print('create Experiments.. ', end='', flush=True)
    for filename in all_files:
      # skip files without suffix ".csv"
      if not filename.endswith('.csv'): continue 
      if not filename.startswith(exp_name): continue
      
      filename_without_extension = filename.replace('.csv', '')
      fields = filename_without_extension.split('_')
      dataset = fields[-2]
      run_id = '_'.join(fields[:-1])
      
      # create Experiment-objects or add files to an existing
      if run_id not in Exp.expDict:
        Exp.expDict[run_id] = Exp(run_id, exp_id, path + filename)
        exp_id += 1
      else:
        Exp.expDict[run_id]._add_file(path + filename)
       
      Exp.dataset_lookup_table.add(dataset)
        
    # read all files
    print(' read files..')
    for experiment in Exp.expDict.values(): experiment._read_files()

