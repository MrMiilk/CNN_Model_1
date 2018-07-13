from CNN_Model_1.model import *
from CNN_Model_1.solver import *


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
    num_epochs = 30
    batch_size = 100
    file_pattern = "dataset\\data-*.TFRecord"

    batch_reader = Reader(file_pattern, num_epochs)
    batch_X, batch_y = batch_reader.input_pipline(batch_size)
    # print(batch_X)
    # print(batch_y)
    # batch_y = tf.reshape(batch_y, [batch_size, -1])
    model = Model(batch_X, batch_y, layers_params, num_classes, batch_size)
    model.train_setting(learing_rate, optimzer, loss_params)
    model.train()