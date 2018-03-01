# Analysis of Catastrophic Forgetting in Deep Neural Networks
* We have analysed 4 types of algorithms to investigate Catastrophic Forgetting:
  * Elastic Weight Consolidation EWC (https://github.com/stokesj/EWC)
  * Fully-Connected NNs (with and without dropout)
  * Convolutional NNs (with and without dropout)
  * Local Winner Takes All  (with and without dropout)

Usage (Linux/bash): source startScripts.bash <model> <parallelruns>
  starts <parallelruns> bash scripts, each executing its part of the experiments
  calls the right python scripts in the right order

analysis: python findBestRun.py <model> realistic|prescient
  takes a bunch if log files that are supposed to be in ./, and spits out the performnance matrix (model/params + task are indices, entry is evaluation result)

visualization; python plotOneExp.py <expID>
  generates a file f.png that visualizes an experimental run. This run can be copied from the results of findBestRun.py


In the tmp subfolder zipsIjcnnLatest are zipfiles with logs that can be used for tests of analysis/visualization, just unzip them first!
