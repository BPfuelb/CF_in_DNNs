

for lr in 0.01 0.001 0.0001 ; do
  # initial training
  python dropout_more_layers.py --train_classes 0 1 2 3 4 --test_classes 0 1 2 3 4 --training_readout_layer 1 --testing_readout_layer -1 --save_model alexConv0${lr} --learning_rate 0.001 --plot_file initTraining${lr}.csv
  # retraining
  python dropout_more_layers.py --train_classes 5 6 7 8 9 --test_classes 0 1 2 3 4 --training_readout_layer 2 --testing_readout_layer -1 --load_model alexConv0${lr} --save_model alexConv1${lr} --learning_rate ${lr} --plot_file retraining${lr}.csv
done



