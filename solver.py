import numpy as np
import tensorflow as tf
from CNN_Model_1.bins import *
from CNN_Model_1.layers import *
from CNN_Model_1.input_data import *


class Solver(object):
    """训练框架"""

    def __init__(self):
        self.y = None
        self.X = None
        self.layers_params = None
        self.num_hidden_layers = None
        self.num_classes = None

        self.pred = None

    def _one_hot(self):
        self.y = tf.one_hot(self.y, self.num_classes)

    def _reset(self):
        self.loss_list = []

    def _loss(self):
        self.loss = loss_layer(self.y, self.pred, self.loss_params)

    def _optimzer(self):
        if self.optimzer_name == 'adam':
            with tf.name_scope('Train'):
                self.optimzer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_step = self.optimzer.minimize(self.loss)
        return

    def _data_batches(self, train_X, train_y, batch_size):
        N = train_X.shape[0]
        num_batches = int((N + batch_size - 1)/batch_size)
        batches = []
        index = tf.random_shuffle(np.arange(N))
        for b in range(num_batches):
            batch_x = train_X[index[b * num_batches: (b + 1) * num_batches]]
            batch_y = train_y[index[b * num_batches: (b + 1) * num_batches]]
            batches.append((batch_x, batch_y))
        return batches

    def train_setting(self, learning_rate,  optimzer, loss_params, print_every=20, verbose=True):
        """未设置：lr_decay"""
        self.optimzer_name = optimzer
        self.learning_rate = learning_rate
        self.loss_params = loss_params
        self.print_every = print_every
        self.verbose = verbose
        ##TODO: 设置损失计算和优化方法##
        self._loss()
        self._optimzer()

    def train(self,):
        """为训练提供接口"""

        print_bool = True if self.print_every is not None else False

        with tf.Session() as sess:
            init_op = tf.group([
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            ])
            sess.run(init_op)
            writer = tf.summary.FileWriter('logs/', sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print('Threads:', threads)
            # foods = {
            #     self.X:batch_X,
            #     self.y:batch_y
            # }
            try:
                for _ in range(1000):
                    if not coord.should_stop():
                        self.train_step.run()
            except tf.errors.OutOfRangeError:
                print('Catch OutOfRangeError')
            finally:
                coord.request_stop()
                print('Finish reading')
            coord.join(threads)


    def check_accuracy(self):
        pass

    def plot_inf(self):
        pass


