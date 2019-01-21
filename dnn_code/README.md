# DNN package

The `dnn.py` include 5 different models: FC, D-FC, CONV, D-CONV and LWTA.
Based on the DNN_MODEL (parser-) parameter the specified DNN is build, trained and evaluated.
The output of the training/testing is based on the given parameters.

## Example

Example call for D-FC experiment: first training and testing (D1D1) (os: Linux) 

```
python3 ./IncrementalLearning/dnn_code/dnn.py 
	--learning_rate=0.01 
	--plot_file /tmp/ExpDist/D-fc_task_D5-5h_lr_0.01_relr_0.001_h1_400_h2_800_h3_200_ds_Devnagari_D1D1.csv 
	--save_model D-fc_task_D5-5h_lr_0.01_relr_0.001_h1_400_h2_800_h3_200_ds_Devnagari_D1D1 
	--test_classes 0 2 3 6 8 
	--train_classes 0 2 3 6 8 
	--dataset_file Devnagari 
	--hidden1 400 --hidden2 800 --hidden3 200 
	--batch_size 100 
	--dropout_hidden 0.5 --dropout_input 0.8 
	--permuteTrain 0 --permuteTrain2 0 --permuteTest 0 --permuteTest2 0 --permuteTest3 0 
	--checkpoints_dir /tmp/ExpDist/checkpoints/
```

## Training
The prepared DNN is trained by the given batch size for $`\mathcal{E}`$ epochs.
The number of iterations per epoch is calculated by the number of training examples of the specific dataset (number of elements in the dataset divided by the batch size).

## Evaluation
The evaluation is also based on the given parameters (multiple measurements for different dataset elements, e.g., a subset of classes)
For each test set the number of batches were calculated to test one epoch.
The average over this accuracies is written to a file, with the combination of the base iteration of training.
The number of measurements points is also set by a parameter (default is 90).
The real number could differ, based on the number of training iterations.
There should be exactly or more then 90 point if possible, else the number of iteration to complete all epochs.