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

def get_train_batch(config):
    filenames = [config['train_data']]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: parser(x, config['num_classes']))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.repeat(100)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def get_train_data(config):
    iterator = get_train_batch(config)
    dataA, label = iterator.get_next()
    return [dataA, label], iterator

def get_test_data(config):
    iterator = get_test_batch(config)
    dataA, label = iterator.get_next()
    return [dataA, label], iterator

def get_test_batch(config):
    filenames = [config['test_data']]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: parser(x, config['num_classes']))
    dataset = dataset.batch(config['batch_size'])
    iterator = dataset.make_initializable_iterator()
    return iterator

def compute_output_matrix(label_max, pred_max, output_matrix):
    for i in xrange(output_matrix.shape[0]):
        temp = pred_max == i
        temp_l = label_max == i
        tp = np.logical_and(temp, temp_l)
        temp[temp_l] = True
        fp = np.logical_xor(temp, temp_l)
        temp = pred_max == i
        temp[fp] = False
        fn = np.logical_xor(temp, temp_l)
        output_matrix[i, 0] += np.sum(tp)
        output_matrix[i, 1] += np.sum(fp)
        output_matrix[i, 2] += np.sum(fn)

    return output_matr