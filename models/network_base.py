
''' AdapNet:  Adaptive  Semantic  Segmentation
              in  Adverse  Environmental  Conditions

 Copyright (C) 2018  Abhinav Valada, Johan Vertens , Ankit Dhall and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

import numpy as np
import tensorflow as tf

class Network(object):
    def __init__(self):
        print 'Network_Construction'

    def _setup(self, data):
        raise NotImplementedError("Implement this method.")

    def _create_loss(self, label):
        raise NotImplementedError("Implement this method.")

    def _create_optimizer(self):
        raise NotImplementedError("Implement this method.")

    def _create_summaries(self):
        raise NotImplementedError("Implement this method.")

    def build_graph(self, data, label=None):
        raise NotImplementedError("Implement this method.")

    def conv2d(self, inputs, kernel_size, stride, out_channels, name=None, padding='SAME'):
        in_channels = inputs.get_shape().as_list()[-1]
        weight_shape = [kernel_size, kernel_size, in_channels, out_channels]
        initializer = tf.contrib.layers.xavier_initializer()

        if self.initializer is 'he':
            n = kernel_size * kernel_size * in_channels
            std = np.sqrt(2.0 / n)
            initializer = tf.truncated_normal_initializer(stddev=std)

        if name is None:
            name = 'weights'

        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('', weight_shape,
                                     initializer=initializer,
                                     trainable=self.training,
                                     dtype=self.float_type,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
                                    )
        return tf.nn.conv2d(inputs, kernel, strides=strides, padding=padding)

    def split_conv2d(self, inputs, kernel_size, rate, out_channels, name=None, padding='SAME', both_atrous=False):
        in_channels = inputs.get_shape().as_list()[-1]
        weight_shape = [kernel_size, kernel_size, in_channels, out_channels]
        initializer = tf.contrib.layers.xavier_initializer()
        if self.initializer is 'he':
            n = kernel_size * kernel_size * in_channels
            std = np.sqrt(2.0 / n)
            initializer = tf.truncated_normal_initializer(stddev=std)
        if name is None:
            name = 'weights'

        strides = [1, 1, 1, 1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('', weight_shape,
                                     initializer=initializer,
                                     trainable=self.training,
                                     dtype=self.float_type,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
                                    )
        kernelA, kernelB = tf.split(kernel, 2, 3)
        if both_atrous:
            outA = tf.nn.atrous_conv2d(inputs, kernelA, rate, padding=padding)
            outB = tf.nn.atrous_conv2d(inputs, kernelB, rate, padding=padding)
        else:
            outA = tf.nn.conv2d(inputs, kernelA, strides=strides, padding=padding)
            outB = tf.nn.atrous_conv2d(inputs, kernelB, rate, padding=padding)
        return tf.concat((outA, outB), 3)

    def batch_norm(self, inputs):
        in_channels = inputs.get_shape().as_list()[-1]

        with tf.variable_scope('BatchNorm'):
            gamma = tf.get_variable('gamma', (in_channels,), initializer=tf.constant_initializer(1.0),
                                    trainable=self.training, dtype=self.float_type,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))
            beta = tf.get_variable('beta', (in_channels,), initializer=tf.constant_initializer(0),
                                   trainable=self.training, dtype=self.float_type,
                                   regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))
            moving_mean = tf.get_variable('moving_mean', (in_channels,),
                                          initializer=tf.constant_initializer(0), trainable=False, dtype=self.float_type)
            moving_var = tf.get_variable('moving_variance', (in_channels,),
                                         initializer=tf.constant_initializer(1), trainable=False, dtype=self.float_type)
