
# train with 0..7
python lwta_more_layers.py --train_classes 0 1 2 3 4 5 6 7 --test_classes 0 1 2 3 4 5 6 7 --learning_rate 0.001  --save_model saad --training_readout_layer 1 --testing_readout_layer 1 --max_steps 2000 --start_at_step 0

# retrain with 8
python lwta_more_layers.py --train_classes 8 --test_classes 0 1 2 3 4 5 6 7 --learning_rate 0.0001 --training_readout_layer 2 --testing_readout_layer 1 --max_steps 2000 --start_at_step 2000 --load_model saad --save_model saadAfterRetr1

# retrain with 9 
python lwta_more_layers.py --train_classes 9 --test_classes 0 1 2 3 4 5 6 7 --learning_rate 0.001 --training_readout_layer 3 --testing_readout_layer 1 --max_steps 2000 --start_at_step 4000 --load_model saadAfterRetr1 --save_model saadAfterRetr2
