
import tensorflow as tf
import numpy as np
import math

# import utils
from .linear_layer import RegLinear, DropLinear

#########################################################################################


class TransferNN(object):

  def __init__(self, node_info, optim=('Adam', 1e-4), name='[tf/NN]', trainable_info=None, keep_prob_info=None,
               flags=None, datasets=None):
    self.name = name
    self.optim = optim
    self.node_info = node_info

    if keep_prob_info == None: keep_prob_info = [0.8] + [0.5] * (len(node_info) - 2)
    self.keep_prob_info = np.array(keep_prob_info)
    self.eval_keep_prob_info = np.array([1.0] * (len(node_info) - 1))

    if trainable_info == None: trainable_info = [True] * len(node_info)
    self.trainable_info = trainable_info

    self.x = tf.placeholder(tf.float32, shape=[None, np.prod(node_info[0])])
    self.y_ = tf.placeholder(tf.float32, shape=[None, node_info[-1]])
    self.drop_rate = tf.placeholder(tf.float32, shape=[len(node_info) - 1])
    
    # added
    self.FLAGS = flags
    self.datasets = datasets

    self._BuildModel()
    self._CrossEntropyPackage(optim)

  def _BuildModel(self):
    h_out_prev = tf.nn.dropout(self.x, self.drop_rate[0])

    self.Layers = []
    self.Layers_dropbase = []
    for l in range(1, len(self.node_info) - 1):
      self.Layers.append(DropLinear(h_out_prev, self.node_info[l], self.drop_rate[l]))
      self.Layers_dropbase.append(self.Layers[-1].dropbase)

      h_out_prev = tf.nn.relu(self.Layers[-1].h_out)

    self.Layers.append(DropLinear(h_out_prev, self.node_info[-1], 1.0))
    self.Layers_dropbase.append(self.Layers[-1].dropbase)
    self.y = self.Layers[-1].h_out

  def _OptimizerPackage(self, obj, optim):
    if optim[0] == 'Adam': return tf.train.AdamOptimizer(optim[1]).minimize(obj)
    elif optim[0] == 'SGD': return tf.train.GradientDescentOptimizer(optim[1]).minimize(obj)
    elif optim[0] == 'Momentum': return tf.train.MomentumOptimizer(optim[1][0], optim[1][1]).minimize(obj)

  def _CrossEntropyPackage(self, optim):
    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
    self.train_step = self._OptimizerPackage(self.cross_entropy, optim)
    self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

  def RegPatch(self, delta):
    self.reg_obj = 0
    self.Layers_reg = []

    for l in range(0, len(self.Layers)):
      self.Layers_reg.append(RegLinear(self.Layers[l]))
      self.reg_obj += delta * self.Layers_reg[l].reg_obj

    cel = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y)
    self.cross_entropy = tf.reduce_mean(cel) + self.reg_obj
    self.train_step = self._OptimizerPackage(self.cross_entropy, self.optim)

  def CalculateFisherMatrix(self, sess, dataset, fisher_batch_size=50):
    """new version of Calculating Fisher Matrix

    Returns:
        FM: consist of [FisherMatrix(layer)] including W and b.
            and Fisher Matrix is dictionary of numpy array
            i.e. Fs[idxOfLayer]['W' or 'b'] -> numpy
    """
    import time
    start = time.time()
    FM = []
    
    batch_size = fisher_batch_size # batch size for fisher accumulation (best is 1!) 
    num_batches = math.ceil(dataset.images.shape[0] / batch_size) # number of batches (round up)  
    
    def feed_dict_fisher(dataset, batch_size=batch_size):
      batch_xs, _ = dataset.next_batch(batch_size)
      return {self.x: batch_xs, self.drop_rate:self.eval_keep_prob_info}
    
    for batch in range(num_batches):
      y_sample = tf.reshape(tf.one_hot(tf.multinomial(self.y, 1), 10), [-1, 10])
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_sample, logits=self.y))
      for l in range(len(self.Layers)):
        if len(FM) < l + 1:
          FM.append({})
          FM[l]['W'] = np.zeros(self.Layers[l].W.get_shape().as_list())
          FM[l]['b'] = np.zeros(self.Layers[l].b.get_shape().as_list())
        W_grad = tf.reduce_sum(tf.square(tf.gradients(cross_entropy, [self.Layers[l].W])), 0)
        b_grad = tf.reduce_sum(tf.square(tf.gradients(cross_entropy, [self.Layers[l].b])), 0)
        W_grad_val, b_grad_val = sess.run([W_grad, b_grad], feed_dict=feed_dict_fisher(dataset))
        FM[l]['W'] += W_grad_val
        FM[l]['b'] += b_grad_val
      print('batch {} of {}'.format(batch + 1, num_batches))

    for l in range(len(self.Layers)):
      FM[l]['W'] += 1e-8
      FM[l]['b'] += 1e-8
    
    print('used time {:.4f}s'.format(time.time() - start))
    return FM
  
  def feed_dict(self, dataset):
    ''' feed dictionary for dataSetOne '''
    batch_size = self.FLAGS.batch_size
    xs, ys = self.datasets[dataset].next_batch(batch_size)
    return {self.x: xs, self.y_: ys, self.drop_rate:self.keep_prob_info}
  
  def Train(self, sess, dataset_train, dataset_test, logTo=None):
    batch_size = self.FLAGS.batch_size
    epochs = self.FLAGS.epochs
    measuring_points = self.FLAGS.measuring_points
    
    num_batches = math.ceil(self.datasets[dataset_train].images.shape[0] / batch_size) # number of batches for one epoch (round up)
    num_batches_all_epochs = int(num_batches * epochs)                                 # number of all batches for all epochs (round)
    test_all_n_batches = num_batches_all_epochs // measuring_points                    # test all n iteration (round) 
    if test_all_n_batches == 0: test_all_n_batches = 1                                 # if less than MEASURING_POINTS iterations for all epochs are done
    
    # train_acc = 0
    for epoch in range(epochs):
      print('epoch number: {}'.format(epoch + 1))
      for batch in range(num_batches):
        # training
        sess.run(self.train_step, feed_dict=self.feed_dict(dataset_train))
        
        # test
        if batch % test_all_n_batches == 0:
          test_acc = self.Test(sess, dataset_test, debug=False)
          
          print ('batch: {} accuracy {}'.format(batch, test_acc))
          if logTo is not None: logTo.write('{},{}\n'.format(batch, test_acc))
      # for batch in epoch
    # for epoch

  def Test(self, sess, dataset, debug=True, logTo=None): 
    test_acc = .0
    batches = math.ceil(self.datasets[dataset].images.shape[0] / self.FLAGS.batch_size) # number of batches for one epoch (round up)
    for _ in range(batches): # test one iteration on dataset from datasets dict 
      test_acc += sess.run(self.accuracy, feed_dict=self.feed_dict(dataset))
    test_acc /= batches
    if debug:             print('test {},{}'.format(dataset, test_acc))
    if logTo is not None: logTo.write('{},{}\n'.format(dataset, test_acc))
    return test_acc

