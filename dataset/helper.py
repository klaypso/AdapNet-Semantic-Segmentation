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
    iterator = dataset.make_one