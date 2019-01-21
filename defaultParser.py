''' provides a parser with default arguments for all experiments 

@author: BPF
'''
import argparse


def printFlags(flags):
    ''' print all flags '''
    side = '-' * int(((80 - len('Flags')) / 2))
    print(side + ' ' + 'Flags' + ' ' + side)
    for k, v, in sorted(vars(flags).items()):
      if v:
        print('\t * {}: {}'.format(k, v))


def create_default_parser():
    ''' create a parser with default parameters for most experiments'''
    parser = argparse.ArgumentParser()

    #===========================================================================
    # Datasets and classes
    #===========================================================================

    #------------------------------------------------------------------- DATASET
    parser.add_argument('--dataset_file', type=str,
                        default='MNIST.pkl.gz',
                        help='''load a compressed pickle file.
                         * MNIST.pkl.gz         #  17MB [209MB extracted]
                         * FashionMNIST.pkl.gz  #  42MB [222MB extracted]
                         * CIFAR10.pkl.gz       # 186MB [739MB extracted]
                         * NotMNIST.pkl.gz      # 108MB [1,7GB extracted]
                         * MADBase.pkl.gz       #  17MB [220MB extracted]
                         * CUB200.pkl.gz        #  20MB [ 69MB extracted] (only 10 classes)
                         * Devnagari.pkl.gz     #  56MB [393MB extracted] (only 10 classes)
                         * EMNIST.pkl.gz        # 309MB [2,8GB extracted] (62 classes)
                         * Fruits.pkl.gz        # 176MB [780MB extracted] (only 10 classes)
                         * Simpsons.pkl.gz      # 122MB [577MB extracted] (only 10 classes)
                         * SVHN.pkl.gz          # 357MB [1.2GB extracted] (only 10 classes)
                        ''')
    
    parser.add_argument('--dataset_dir', type=str,
                        default='./IncrementalLearning/',
                        help='Directory for storing input data (pkl.gz files).')

    #------------------------------------------------------------------- CLASSES
    parser.add_argument('--train_classes', type=int, nargs='*',
                        help="Take only the specified Train classes from dataSet")
    parser.add_argument('--train2_classes', type=int, nargs='*',
                        help="Take only the specified Train2 classes from dataSet")

    parser.add_argument('--test_classes', type=int, nargs='*',
                        help="Take the specified Test classes from dataSet")
    parser.add_argument('--test2_classes', type=int, nargs='*',
                        help="Take the specified Test classes from dataSet. No test if empty")
    parser.add_argument('--test3_classes', type=int, nargs='*',
                        help="Take the specified Test classes from dataSet. No test3 if empty")

    #-------------------------------------------------------------- PERMUTE DATA
    parser.add_argument('--permuteTrain', type=int,
                        default=0,
                        help='Provide random seed for permutation train. default: no permutation')
    parser.add_argument('--permuteTrain2', type=int,
                        default=0,
                        help='Provide random seed for permutation train2. default: no permutation')
    parser.add_argument('--permuteTest', type=int,
                        default=0,
                        help='Provide random seed for permutation test.  default: no permutation')
    parser.add_argument('--permuteTest2', type=int,
                        default=0,
                        help='Provide random seed for permutation test2.  default: no permutation')
    parser.add_argument('--permuteTest3', type=int,
                        default=0,
                        help='Provide random seed for permutation test3.  default: no permutation')

    parser.add_argument('--mergeTrainInto', type=int, nargs='+',
                        default=-1,
                        help='merge train set arg into train set arg2')

    parser.add_argument('--mergeTestInto', type=int, nargs='+',
                        default=-1,
                        help='merge test set arg into train set arg2')

    #===========================================================================
    # DNN configuration 
    #===========================================================================

    #-------------------------------------------------------- DROPTOUT PARAMETER
    parser.add_argument('--dropout_hidden', type=float,
                        default=1,
                        help='Keep probability for dropout on hidden units.')
    parser.add_argument('--dropout_input', type=float,
                        default=1,
                        help='Keep probability for dropout on input units.')

    #----------------------------------------------------------- LAYER PARAMETER
    parser.add_argument('--hidden1', type=int,
                        default=800,
                        help='Number of hidden units in layer 1')
    parser.add_argument('--hidden2', type=int,
                        default=800,
                        help='Number of hidden units in layer 2')
    parser.add_argument('--hidden3', type=int,
                        default=800,
                        help='Number of hidden units in layer 3')

    #----------------------------------- BATCH_SIZE, LEARNING_RATE, EPOCHS, ETC.
    parser.add_argument('--batch_size', type=int,
                        default=100,
                        help='Size of mini-batches we feed from train dataSet.')
    parser.add_argument('--test_batch_size', type=int,
                        default=100,
                        help='Size of mini-batches we feed from test dataSet.')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--measuring_points', type=int,
                        default=90,
                        help='number of tests over all epochs')

    parser.add_argument('--epochs', type=int,
                        default=10,
                        help='the number of training epochs per task, number of iterations will be calculated based on training datset size')

    #----------------------------------------------------------- LOAD/SAVE MODEL, CLEANUP
    parser.add_argument('--load_model', type=str,
                        help='Load previously saved model. Leave empty if no model exists.')
    parser.add_argument('--save_model', type=str,
                        help='Provide path to save model.')
    
    #---------------------------------------------------------- MERGE DATASETS
    parser.add_argument('--mergeTest12', type=eval,
                        default=False,
                        help='merge sets test and test2 to form test3?')
    parser.add_argument('--mergeTrainWithPermutation', type=eval,
                        default=False,
                        help='merge train  set and permuted train set?')
    
    #---------------------------------------------------------- DIRECTORY PATHES
    parser.add_argument('--data_dir', type=str,
                        default='./',
                        help='Directory for storing input data')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='%TMP%/ExpDist/checkpoints',
                        help='Checkpoints log directory')
    
    #----------------------------------------------------- CSV OUTPUTS FOR PLOTS
    parser.add_argument('--plot_file', type=str,
                        default=None,
                        help='Filename for csv file to plot. Give .csv extension after file name.')
    parser.add_argument('--plot_file2', type=str,
                        default=None,
                        help='Filename for csv file to plot. Give .csv extension after file name.')
    parser.add_argument('--plot_file3', type=str,
                        default=None,
                        help='Filename for csv file to plot3. Give .csv extension after file name.')
    
    return parser
