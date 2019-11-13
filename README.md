# A comprehensive, application-oriented study of catastrophic forgetting in DNNs

This repository contains the code base for models, experiment creation and evaluation. 

Dependencies: python3, numpy, tensorflow 1.7, scipy, matplotlib, and more...

PYTHONPATH: must be set to absolute path of main directory, e.g.
`export PYTHONPATH=$PYTHONPATH:$(pwd)`

Used models: (D-)FC, (D-)CONV, LWTA, EWC, IMM


# Usage
1. create a experiment batch file with the doExperiments script, e.g., `python3 doExperiments.py --exp FC`
2. process all created experiments (os depended parameters like pathes etc.)
3. use the `experiment_processor.py` to do a model selection based on the results (csv files) of the training phase, e.g., `python3 experiment_processor --path <dir with experiment output files> --evalMode realistic_first`
4. process the recreated experiments for re-training phase
5. evaluate the output of the re-trained models with `python3 experiment_processor --path <dir with experiment output files> --evalMode realistic_second`

# Structure

Rough overview of the repository:

 * [Dataset](./dataset/README.md) contains code to recreate the datasets for the experiments
 * [Eval](./eval/README.md) contains code for evaluation of the outputs of the experiments (an recreate experiments for second run)
 * [DNN_code](./dnn_code/README.md) contains code for the models: (D-)FC, (D-)CONV and LWTA
 * [EWC_code](./ewc_code/README.md) contains code for the EWC model
 * [IMM_code](./imm_code/README.md) contains code for the IMM model
 
## defaultParser.py

Contains the default parser which is used by all experiments. 
Model specific parameters could be added by experiment.

## doExperiments.py

This code create batch files which could be processed singly if the local mode (in the source code) is activated. 
Otherwise an operating system independent batch file is created which we process distributed by our own distribution system (not public available). 
This script is also used to recreate the experiments for the re-training phase.
Therefore the fixed model parameters were passed as additional parameter.
From this, the varied re-train parameters, respectively the experiements with the hyper-parameters were created as new experiment batch file.   

 
 # Catastrophic forgetting: still a problem for DNNs
 Paper: https://arxiv.org/abs/1905.08077 (Tag v2.0)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  
