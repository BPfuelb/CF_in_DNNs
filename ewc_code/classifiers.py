
import os
import tensorflow as tf
import csv
import math

from ewc_code.network import Network


class Classifier(Network):
  """Supplies fully connected prediction model with training loop which absorbs minibatches and updates weights."""

  def __init__(self, checkpoint_path='logs/checkpoints/', summaries_path='logs/summaries/', flags=None, *args, **kwargs):
    super(Classifier, self).__init__(*args, **kwargs)
    self.checkpoint_path = checkpoint_path
    self.summaries_path = summaries_path
    
    self.merged = None
    self.optimizer = None
    self.train_step = None
    self.accuracy = None
    self.loss = None
    
    self.FLAGS = flags
    
    self.create_loss_and_accuracy()

  def create_feed_dict(self, dataset, keep_hidden=0.5, keep_input=0.8):
    ''' create a feed dict form given dataset
    
    @param dataset: dataset to get next batch (xs, ys)
    @param keep_hidden: probability for dropout in hidden layer
    @param keep_input: probability for dropout in input layer
    @return: feed dict {xs, ys, keep_hidden, keep_input}
    '''
    batch_xs, batch_ys = dataset.next_batch(self.FLAGS.batch_size)
    feed_dict = {self.x: batch_xs, self.y: batch_ys}
    if self.apply_dropout:
      feed_dict.update({self.keep_prob_hidden: keep_hidden, self.keep_prob_input: keep_input})
      
    # TODO: if dataset_test: no dropout! # currently in evaluate_mod() method
    # if self.apply_dropout and dataset == ?: 
    #   feed_dict_test.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0}) # overwrite dropout for testing
    return feed_dict
  
  def train_mod(self, sess, model_name, model_init_name, dataset, fisher_multiplier,
                learning_rate, testing_data_sets, plot_files=["ewc.csv"]):  
    print('training {} with weights initialized at {}'.format(model_name, model_init_name))
    self.prepare_for_training(sess, model_name, model_init_name, fisher_multiplier, learning_rate)
    self.save_weights(-1, sess, model_name)

    writers = [csv.writer(open(plot_file, 'w')) for plot_file in plot_files if plot_file is not None]
    
    num_batches = math.ceil(dataset.images.shape[0] / self.FLAGS.batch_size)      # number of batches for one epoch (round up) 
    num_batches_all_epochs = int(num_batches * self.FLAGS.epochs)                 # number of all batches for all epochs (round)
    test_all_n_iterations = num_batches_all_epochs // self.FLAGS.measuring_points # test all n iteration (round)
    if test_all_n_iterations == 0: test_all_n_iterations = 1                      # if less than MEASURING_POINTS iterations for all epochs are done
    
    for epoch in range(self.FLAGS.epochs):
      print('epoch: {}'.format(epoch + 1))
      for i in range(num_batches):
        iteration = i + epoch * num_batches # calculate the iteration
        self.minibatch_sgd_mod(sess, i, dataset, test_all_n_iterations, testing_data_sets, writers)
    
    import time
    start = time.time()
    print('start calculating fisher matrix')
    self.update_fisher_full_batch(sess, dataset) # FIXME: do not calculate for D1D1, baseline
    print('end used time {:.2f}s'.format(time.time() - start))
    
    self.save_weights(iteration, sess, model_name)
    print('finished training {}'.format(model_name))
  
  def minibatch_sgd_mod(self, sess, i, dataset, log_frequency, testing_data_sets, csv_writers):
    ''' 
    @param i: iteration (int)
    @param log_frequency: test all n iterations (int)
    '''
    feed_dict = self.create_feed_dict(dataset)
    sess.run(self.train_step, feed_dict=feed_dict)

    j = 0
    for testing_data_set in testing_data_sets:
      if testing_data_set is None: continue
      if j >= len(csv_writers): continue

      if log_frequency and i % log_frequency is 0:
        self.evaluate_mod(sess, i, testing_data_set, csv_writers[j])
      j += 1

  def evaluate_mod(self, sess, iteration, testing_data_set, csv_writer):
    feed_dict_test = self.create_feed_dict(testing_data_set, keep_hidden=1.0, keep_input=1.0)
    
    num_batches_test_epoch = int(testing_data_set.images.shape[0] / self.FLAGS.batch_size + .5)  # number of batches for one epoch (round up + .5)
    if num_batches_test_epoch == 0: num_batches_test_epoch = 1                                   # if less than MEASURING_POINTS iterations for all epochs are done
    acc_all = 0.0
    for _ in range(num_batches_test_epoch):
      if self.apply_dropout: feed_dict_test.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0}) # overwrite dropout for testing
      acc_all += sess.run([self.merged, self.accuracy], feed_dict=feed_dict_test)[1]
    acc_all /= num_batches_test_epoch
    print('Accuracy at step {} is: {}'.format(iteration, acc_all))
    csv_writer.writerow([iteration, acc_all])

  def update_fisher_full_batch(self, sess, dataset):
    dataset._index_in_epoch = 0  # ensures that all training examples are included without repetitions
    sess.run(self.fisher_zero_op)
    
    batch_size = self.ewc_batch_size # batch size for fisher accumulation (best is 1!) 
    num_ewc_batches = math.ceil(dataset.images.shape[0] / batch_size) # number of batches (round up) 
    
    def feed_dict_fisher(dataset, batch_size=batch_size):
      batch_xs, batch_ys = dataset.next_batch(batch_size)
      return {self.x_fisher: batch_xs, self.y_fisher: batch_ys}
  
    # accumulate fisher matrix
    for _ in range(num_ewc_batches): 
      sess.run(self.fisher_accumulate_op, feed_dict=feed_dict_fisher(dataset))
    
    sess.run(self.fisher_full_batch_average_op)
    sess.run(self.update_theta_op)

  def prepare_for_training(self, sess, model_name, model_init_name, fisher_multiplier, learning_rate):
    # self.writer = tf.summary.FileWriter(self.summaries_path + model_name, sess.graph)
    self.merged = tf.summary.merge_all()
    self.train_step = self.create_train_step(fisher_multiplier if model_init_name else 0.0, learning_rate)
    init = tf.global_variables_initializer()
    sess.run(init)
    if model_init_name:
      print ("Loading model: " + model_init_name)
      self.restore_model(sess, model_init_name)

  def create_loss_and_accuracy(self):
    with tf.name_scope("loss"):
      average_nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.y))  # optimized
      tf.summary.scalar("loss", average_nll)
      self.loss = average_nll
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
      tf.summary.scalar('accuracy', accuracy)
      self.accuracy = accuracy

  def create_train_step(self, fisher_multiplier, learning_rate):
    with tf.name_scope("optimizer"):
      self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      penalty = tf.add_n([tf.reduce_sum(tf.square(w1 - w2) * f) for w1, w2, f
                          in zip(self.theta, self.theta_lagged, self.fisher_diagonal)])
      return self.optimizer.minimize(self.loss + (fisher_multiplier / 2) * penalty, var_list=self.theta)

  def save_weights(self, time_step, sess, model_name):
    if model_name is None or model_name == "": return
    if self.FLAGS.save_model is None:          return 
    
    if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)
    self.saver.save(sess=sess, save_path=self.checkpoint_path + model_name + '.ckpt')
    print('saving model ' + self.checkpoint_path + model_name + '.ckpt at time step ' + str(time_step))

  def restore_model(self, sess, model_name):
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_path, latest_filename=model_name)

    print ("loading..", self.checkpoint_path + "/" + model_name + ".ckpt")
    self.saver.restore(sess=sess, save_path=self.checkpoint_path + model_name + ".ckpt")

