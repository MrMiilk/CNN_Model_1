import numpy as np
import tensorflow as tf
from Example_Model.bins import *
from Example_Model.model import *
from Example_Model.solver import *


if __name__ == '__main__':
    layers_params = [
        ['conv', [([7, 7, 3, 32], [32]), ('relu', [1,2,2,1], 'VALID')]],
        ['flat', [5408]],
        ['af', [([5408, 10], [10]), ('', )]]
    ]
    num_classes = 10
    learing_rate = 3e-3
    optimzer = 'adam'
    loss_params = ['hinge_loss']

    X, y = create_placeholder()

    model = Model(X, y, layers_params, num_classes)
    model.train_setting(learing_rate, optimzer, loss_params)
    model.train(None, None)