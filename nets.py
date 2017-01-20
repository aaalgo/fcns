#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1


# conv2d and conv2d_transpose

# conv2d output size if padding = 'SAME':   W <- (W + S -1)/S 
#                                 'VALID':  W <- (W - F + S)/S
def simple (X, num_classes=2):
    # stride is  2 * 2 * 2 * 2 = 16
    net = X
    layers = [X]
    with tf.name_scope('simple'), slim.arg_scope([slim.max_pool2d], padding='SAME'):
        # slim.arg_scope([slim.conv2d]):
        # slim.conv2d defaults:
        #   padding = 'SAME'
        #   activation_fn = nn.relu
        # parameters: net, out_channels, kernel_size, stride
        net = slim.conv2d(net, 100, 5, 2, scope='conv1')
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.conv2d(net, 200, 5, 2, scope='conv2')
        net = slim.max_pool2d(net, 2, 2, scope='pool2')
        net = slim.conv2d(net, 300, 3, 1, scope='conv3')
        net = slim.conv2d(net, 300, 3, 1, scope='conv4')
        net = slim.dropout(net, keep_prob=0.9, scope='dropout')
        net = slim.conv2d(net, 20, 1, 1, scope='layer5')
        net = slim.conv2d_transpose(net, num_classes, 31, 16, scope='upscale')
    net = identity(net, 'logits')
    return net, 16

def  resnet_v1_50 (X, num_classes=2):
    with tf.name_scope('resnet_v1_50':
        net, _ = resnet_v1.resnet_v1_50(X,
                                num_classes=num_classes,
                                global_pool = False,
                                output_stride = 16)
        net = slim.conv2d_transpose(net, num_classes, 31, 16, scope='upscale')
    net = identity(net, 'logits')
    return net, 16

