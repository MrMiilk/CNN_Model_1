import numpy as np
import tensorflow as tf
from CNN_Model_1.bins import *


def gauss_initer():
    with tf.name_scope('Gauss_initer'):
        initer = tf.random_normal_initializer()
    return initer


def affine(X, param, layer_index, initializer):
    """全连接层"""
    variable_params, layer_params = parse_param('af', param)
    w_shape, b_shape = variable_params
    nolinear_func = layer_params[0]

    with tf.name_scope('Layer_AF' + str(layer_index)):
        with tf.variable_scope('Variables', reuse=tf.AUTO_REUSE):
            weight = tf.get_variable(name='weight' + str(layer_index), shape=w_shape, initializer=initializer)
            bias = tf.get_variable(name='bias' + str(layer_index), shape=b_shape, initializer=initializer)
        output = tf.add(tf.matmul(X, weight), bias)
        if hasattr(tf.nn, nolinear_func):
            with tf.name_scope(nolinear_func):
                nl_func = getattr(tf.nn, nolinear_func)
                output = nl_func(output)
    return output


def conv(X, param, layer_index, initializer):
    """卷积层"""
    variable_params, layer_params = parse_param('conv', param)
    filter_shape, b_shape = variable_params
    nolinear_func, stride, pad = layer_params

    with tf.name_scope('Layer_Conv' + str(layer_index)):
        with tf.variable_scope('Variables', reuse=tf.AUTO_REUSE):
            filter = tf.get_variable(name='filter' + str(layer_index), shape=filter_shape, initializer=initializer)
            bias = tf.get_variable(name='bias' + str(layer_index), shape=b_shape, initializer=initializer)
        output = tf.nn.conv2d(X, filter, stride, pad) + bias
        if hasattr(tf.nn, nolinear_func):
            with tf.name_scope(nolinear_func):
                nl_func = getattr(tf.nn, nolinear_func)
                output = nl_func(output)
    return output


def flat(X, param, layer_index):
    with tf.name_scope('Layer_flat' + str(layer_index)):
        len = parse_param('flat', param)
        output = tf.reshape(X, [-1, len])
    return output


def loss_layer(y, y_out, param):
    loss_name = parse_param('loss', param)
    if loss_name == 'hinge_loss':
        total_loss = tf.losses.hinge_loss(y, logits=y_out)
        mean_loss = tf.reduce_mean(total_loss)
        return mean_loss
    else:
        raise AttributeError("Don't support this kind of loss")