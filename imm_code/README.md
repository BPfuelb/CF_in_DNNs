# IMM package

The initial file to train an IMM net is `imm.py`. Source: https://arxiv.org/abs/1703.08475

```
@article{DBLP:journals/corr/LeeKHZ17,
  author    = {Sang{-}Woo Lee and
               Jin{-}Hwa Kim and
               JungWoo Ha and
               Byoung{-}Tak Zhang},
  title     = {Overcoming Catastrophic Forgetting by Incremental Moment Matching},
  journal   = {CoRR},
  volume    = {abs/1703.08475},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.08475},
  archivePrefix = {arXiv},
  eprint    = {1703.08475},
  timestamp = {Mon, 13 Aug 2018 16:47:45 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LeeKHZ17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Source code from: https://github.com/btjhjeon/IMM_tensorflow
Customizations:
 * add measuring points during training/retraining
 * loading own datasets
 * adding parameters
 * ...
 
## Example

Example call for IMM experiment: training and testing on both tasks (baseline) (os: Linux) 

```
python3 ./IncrementalLearning/imm_code/imm.py 
	--learning_rate 0.01 --learning_rate2 0.001 
	--mean_imm False --mode_imm False 
	--optimizer SGD 
	--plot_file /tmp/ExpDist/l2tIMM_task_D5-5b_lr_0.01_relr_0.001_h1_400_h2_200_h3_0_ds_Fruits_baseline.csv 
	--tasks 0 
	--train_classes 0 2 4 6 8	--train2_classes 1 3 5 7 9 
	--test_classes 0 2 4 6 8 	--test2_classes 1 3 5 7 9 
	--mergeTestInto 2 1 --mergeTrainInto 2 1 
	--dataset_file Fruits.pkl.gz 
	--hidden1 400 --hidden2 200 --hidden3 -1 
	--batch_size 100 
	--dropout_hidden 0.5 --dropout_input 0.8 
	--permuteTrain 0 --permuteTrain2 0 --permuteTest 0 --permuteTest2 0 --permuteTest3 0 
	--checkpoints_dir /tmp/ExpDist/checkpoints/
```