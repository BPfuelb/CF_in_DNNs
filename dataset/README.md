# Datasets 

We provide a semi automatic construction method for different datasets.

* [CIFAR-10](cifar10.py)
* [CUB200](cub200.py)
* [Devnagari](devnagari.py)
* [EMNIST](emnist.py)
* [FashionMNIST](fashionmnist.py)
* [FRUITS](fruits.py)
* [MADBase](madbase.py)
* [MNIST](mnist.py)
* [notMNIST](notmnist.py)
* [SIMPSONS](simpsons.py)
* [SVHN](svhn.py)

Each dataset class inherit from the abstract base class `Dataset` which provide abstract properties, abstract methods and default implementations for the creation of datasets. 

The abstract base class `Dataset` provide a template pattern to create a dataset step by step (if the `load_dataset` method is invoked by a model).

1. if the dataset was already prepared (existing pickle-file), it will be loaded.
 
Otherwise:
2. the `download_data` method is invoked
3. the `extract_data` method is invoked
4. the `prepare_data` method is invoked, all data from self.data_train, self.labels_train, self.data_test and self.labels_test were used for further steps
5. if no explicit test data exists, a percent value (`SPLIT_TRAINING_TEST` default is 10%) of the training data were extracted and stored as test data.
6. training and test data were shuffled (seeded)
7. data_train and data_test were flattened
8. normalization is applied to data_train and data_test. The VALUE_RANGE is used for normalization
9. the labels for training and testing were converted to one-hot-vector
10. the converted data were stored in a pickle file as a dictionary: 
        'data_train':   data_train,
        'labels_train': labels_train,
        'data_test':    data_test,
        'labels_test':  labels_test,
        'properties':   properties
    }
11. the `clean_up` method is invoked
12. if the inherited dataset class do not overwrite the validate_shape property, the validation mechanism is invoked after the dataset is restored from disk. 
13. the loaded data get prepared by the given parameter in the FLAG variable, e.g., permute data, exclude classes. 
Further the dataset is converted to the DataSet class provided by TensorFlow.

### An example is given for fashionmnist:

Following abstract properties have to be overwritten by the dataset class: 
 * `name` (name of the dataset, should be the same name as the class)
 * `shape` (shape of the input data, will be used for reshaping)
 * `pickle_file` (the pickle filename were the data are stored after the transformation)
 * `num_train` (number of training elements, should match for verification)
 * `num_test` (number of test elements, should match for verification)
 * `num_classes` (number of classes, should match for verification)
   
Following abstract methods must be implemented (could be empty if not needed)
 * `download_data` (download the needed files for further preparation of the data, default download mechanisms get provided by the Downloader class)
 * `extract_data` (decompress and load data into memory, default decompress mechanism is provided by Data_Loader class)
 * `prepare_data` convert method for data in memory, must store data in self.data_train, self.labels_train, self.data_test and self.labels_test for further steps
 * `clean_up` (remove not needed files for dataset creation)
 