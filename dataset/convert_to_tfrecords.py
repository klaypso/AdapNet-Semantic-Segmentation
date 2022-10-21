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

import argparse
import cv2
import numpy as np
import tensorflow as tf

def _int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))

def _bytes_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-f', '--file')
PARSER.add_argument('-r', '--record')
PARSER.add_argument('-m', '--mean')

def decode(txt):
    with open(txt) as file_handler:
        all_list = file_handler.readlines()

    file_list = []
    for line in all_list:
        temp = line.strip('\n').split(' ')
        file_list.append(temp)

    return file_list

def convert(f, record_name, mean_flag):
    count = 0.0
    writer = tf.python_io.TFRecordWriter(record_name)

    if mean_flag:
        mean = np.zeros(cv2.imread(f[0][0]).shape, np.float32)

    for name in f:
        modality1 = cv2.imread(name[0])
     