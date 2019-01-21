# EWC package

Contains the Elastic Weight Consolidation (EWC) code base.
The initial file to train an IMM net is `ewc_with_options.py`. Source: https://arxiv.org/abs/1612.00796

```
@article{DBLP:journals/corr/KirkpatrickPRVD16,
  author    = {James Kirkpatrick and
               Razvan Pascanu and
               Neil C. Rabinowitz and
               Joel Veness and
               Guillaume Desjardins and
               Andrei A. Rusu and
               Kieran Milan and
               John Quan and
               Tiago Ramalho and
               Agnieszka Grabska{-}Barwinska and
               Demis Hassabis and
               Claudia Clopath and
               Dharshan Kumaran and
               Raia Hadsell},
  title     = {Overcoming catastrophic forgetting in neural networks},
  journal   = {CoRR},
  volume    = {abs/1612.00796},
  year      = {2016},
  url       = {http://arxiv.org/abs/1612.00796},
  archivePrefix = {arXiv},
  eprint    = {1612.00796},
  timestamp = {Mon, 13 Aug 2018 16:46:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/KirkpatrickPRVD16},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Source code from: https://github.com/stokesj/EWC 
Customizations:
 * add measuring points during training/retraining
 * loading own datasets
 * adding parameters
 * ...

## Example

Example call for EWC experiment: first training and testing (D1D1) (os: Linux) 

```
python3 ./IncrementalLearning/ewc_code/ewc_with_options.py 
	--learning_rate=0.01 
	--plot_file /tmp/ExpDist/EWC_task_D5-5a_lr_0.01_relr_0.001_h1_200_h2_400_h3_400_ds_Fruits_D1D1.csv 
	--test_classes 0 1 2 3 4 
	--train_classes 0 1 2 3 4 
	--dataset_file Fruits.pkl.gz 
	--hidden1 200 --hidden2 400 --hidden3 400 
	--batch_size 100 
	--dropout_hidden 1.0 --dropout_input 1.0 
	--permuteTrain 0 --permuteTrain2 0 --permuteTest 0 --permuteTest2 0 --permuteTest3 0 
	--checkpoints_dir /tmp/ExpDist/checkpoints/ 
```
 
## Alternatives: 
 * https://github.com/ariseff/overcoming-catastrophic
 * https://github.com/yashkant/Elastic-Weight-Consolidation
 * https://github.com/vield/less-forgetful-nns
 * https://github.com/moskomule/ewc.pytorch
 * https://github.com/kuc2477/pytorch-ewc