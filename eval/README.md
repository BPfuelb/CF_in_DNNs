# Eval Package

The evaluation package use the output files (csv-files) from the experiments and create different plots for the "best" parameter for various models.

e.g. `python3 experiments_processor.py --evalMode realistic_second --path ./fc`

## Parameter 

Evaluation strategies (parameter `--evalMode`):
 * `realistic_fist` first run, only for model selection on D1, recreate experiments for retraining
 * `realistic_second` evaluation of the retraining
 * `imm_search` evaluates imm experiments based on the best accuracy for mode and mean (best $\alpha$)
 * `prescient` (deprecated) model selection based on all (training on both tasks) processed experiments
 * `realistic` (deprecated) only used if all experiments were pre-processed

Path to csv-files (parameter `--path`):
 
### Optional parameter

To exclude different tasks, models or dataset the following three parameters can be used: 
 * `--task` only evaluate a specific task. 
 * `--modelID` only evaluate a specific model (by id).
 * `--dataset` only evaluate a specific dataset.

Default value is `None`: no filter &rarr; for all tasks, models and datasets. 

## Evaluation process 

1. The first step for all strategies is to read the outputs of the experiments (csv-files). 
This functionality is provided by the `Experiment` class.
2. Based on the chosen evaluation strategy the output is:
   * Unordered sub-list.
   * plotted in 2D 
   * plotted in 3D 
   * LaTeX table
   * retraining experiments were generated via `doExperiments.py`
The output must be enabled or disabled in the specific function in the code. 

## Rough description of code

The `experiments_processor.py` file is the entry point for evaluation. 
The first step is to create `Experiment` objects (`experiments.py`) by reading the output csv files from each experiment.
The results were aggregated by maximum and the best hyperparmeters determined (`realistic_fist`). 
Output of the first run is the recreation of experiments with `doExperiments.py`, where only the parameters for the retraining were varied.
The `realistic_second` method use the csv files as input and search for the best experiments (hyperparameter) again and plot these via `experiment_processor.py` (or `Experiment` object for single plots). 