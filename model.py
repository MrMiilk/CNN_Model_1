import tensorflow as tf
from CNN_Model_1.bins import *
from CNN_Model_1.layers import *
from CNN_Model_1.solver import *


class Model(Solver):
    """继承Solver类"""

    def __init__(self, X, y, layers_params, num_classes, batch_size):
        """初始化模型"""
        super(Model, self).__init__()
        self.X = X
        self.y = y
        self.layers_params = layers_params
        self.num_hidden_layers = len(layers_params)
        self.num_classes = num_classes
        self.batch_size = batch_size

        self._one_hot()
        self._reset()
        self._forward_model()

    def _forward_model(self):
        """搭建模型"""
        a = self.X
        initer = gauss_initer()
        for layer_index in range(1, self.num_hidden_layers + 1):
            layer_type, param = self.layers_params[layer_index - 1]
            if layer_type == 'af':
                a = affine(a, param, layer_index, initer)
            if layer_type == 'conv':
                a = conv(a, param, layer_index, initer)
            if layer_type == 'flat':
                a = flat(a, param, layer_index)
        self.pred = a#tf.reshape(a, [self.batch_size, 1, -1])
