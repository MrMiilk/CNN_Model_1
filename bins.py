import tensorflow as tf


def create_placeholder():
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        Y = tf.placeholder(dtype=tf.int64, shape=[None])
    return X, Y


def parse_param(func_name, param_bag):
    ##TODO: parse parameters for layers##
    if func_name == 'conv':
        variable_params, layer_params = param_bag
        return variable_params, layer_params
    if func_name == 'af':
        variable_params, layer_params = param_bag
        return variable_params, layer_params
    if func_name == 'flat':
        return param_bag[0]
    if func_name == 'loss':
        return param_bag[0]