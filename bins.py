import tensorflow as tf


def create_placeholder():
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        Y = tf.placeholder(dtype=tf.int64, shape=[None, 1])
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

##两个数据读取的辅助函数
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))