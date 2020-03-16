# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates a single trained model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf  # tf

from evanet import model_dna

flags.DEFINE_string('checkpoints', '',
                    'Comma separated list of model checkpoints.')

FLAGS = flags.FLAGS

# encoded protobuf strings representing the final RGB and optical flow networks
_RGB_NETWORKS = [
    'CAMQARgCIAEoATAJOhQIAhAAGggIARAAGAEgARoECAMYBzomCAQQABoECAMYAxoCCAAaCAgBEAAYASABGgIIABoICAEQABgFIAE6HggFEAAaBAgDGAcaAggAGggIARABGAsgARoECAMYCzoeCAMQABoECAMYCxoICAEQABgBIAEaCAgBEAEYAyAB',
    'CAMQARgCIAEoAzAJOioIAhAAGgQIAxgBGgQIAxgDGg4IAhABGAEgASgJMAE4AhoECAMYARoCCAA6RAgDEAAaBAgDGAEaBAgDGAcaDggCEAIYBSABKAEwATgAGg4IAhABGAEgASgLMAE4AhoCCAAaDggCEAIYCyABKAEwATgAOhQIBRAAGgQIAxgFGgIIABoECAMYCToUCAQQABoICAEQAhgBIAEaBAgDGAU='
]
_FLOW_NETWORKS = [
    'CAMQARgCIAEoATAJOhQIAhAAGggIARAAGAEgARoECAMYBzomCAQQABoECAMYAxoCCAAaCAgBEAAYASABGgIIABoICAEQABgFIAE6HggFEAAaBAgDGAcaAggAGggIARABGAsgARoECAMYCzoeCAMQABoECAMYCxoICAEQABgBIAEaCAgBEAEYAyAB',
    'CAMQARgCIAEoAzAJOioIAhAAGgQIAxgBGgQIAxgDGg4IAhABGAEgASgJMAE4AhoECAMYARoCCAA6RAgDEAAaBAgDGAEaBAgDGAcaDggCEAIYBSABKAEwATgAGg4IAhABGAEgASgLMAE4AhoCCAAaDggCEAIYCyABKAEwATgAOhQIBRAAGgQIAxgFGgIIABoECAMYCToUCAQQABoICAEQAhgBIAEaBAgDGAU='
]


def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


def main(_):
  weights = [x for x in FLAGS.checkpoints.split(',') if len(x)]
  videos = [
      'evanet/data/v_CricketShot_g04_c01_rgb.npy',
      # 'evanet/data/p1_backhand_s1.npy'
  ]
  # f=open('filenames.txt','r')
  # videos=f.read().split('\n')

  label_map = 'evanet/data/label_map.txt'
  kinetics_classes = [x.strip() for x in open(label_map)]

  # create model
  final_logits = np.zeros((400,), np.float32)
  for i, model_str in enumerate(_RGB_NETWORKS):
    tf.reset_default_graph()
    #vid = np.load(videos[i])
    vid_placeholder = tf.placeholder(tf.float32, shape=(1,None,224,224,3))#, shape=vid.shape

    model = model_dna.ModelDNA(base64.b64decode(model_str), num_classes=6)
    endpoints = model.model(vid_placeholder, mode='eval',only_endpoints=True,final_endpoint='LastCell')
    #logits = model.model(vid_placeholder, mode='eval')

    variable_map = {}
    for var in tf.global_variables():
      variable_map[var.name.replace(':0', '').replace('Conv/', 'Conv3d/')] = var

    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    with tf.Session() as sess:
      if i <= len(weights) - 1:
        saver.restore(sess,
                      os.path.join('evanet', 'data', 'checkpoints', weights[i]))
      else:
        print('Warning, model has no pretrained weights')
        sess.run(tf.global_variables_initializer())
      for j in range(len(videos)):
          vid = np.load(videos[j])
          print("Shape is ",vid.shape)
          #print("Value is :",vid)
          out_endpoints = sess.run([endpoints], feed_dict={vid_placeholder: vid})
          if not os.path.isdir(os.path.join('evanet','data',str(i))):
              os.mkdir(os.path.join('evanet','data',str(i)))
          path = os.path.join('output' + str(i) + '_features')
          # path = os.path.join('evanet','data',str(i+1),videos[j].split('.npy')[0].split('/')[-1]+'_'+str(i)+ '_features')
          np.save(path,out_endpoints[0]['LastCell'])
      #out_logits = sess.run([logits], feed_dict={vid_placeholder: vid})
      #print(type(out_logits[0])," is out logits type")
      #print(out_logits[0].shape," is out logits shape")
      #final_logits += out_logits[0][0]

  # average prediction
  """
  print("************Engering For Loop*************")
  for i, model_str in enumerate(_RGB_NETWORKS):
      filepath=os.path.join(videos[i].split('.npy')[0],str(i)+'_-features.npy')
      print(i,filepath)
      if os.path.isfile(filepath):
          print("loading file")
          #seq= np.load(filepath).mean(axis=(1,2,3))
          #final_logits+=seq[0]
  
          
  
  #print(type(out_logits[0])," is out logits type in the end")
  #print(out_logits[0].shape," is out logits shape in the end")
  final_logits /= float(len(_RGB_NETWORKS))
  print(final_logits)
  print(type(final_logits), " is type")
  print(final_logits.shape, " is shape")
  
  predictions = softmax(final_logits)
  sorted_indices = np.argsort(final_logits)[::-1]
  #tf.io.write_file('test_endpoints',endpoints['LastBeforeLogits'])
  #print(out_endpoints[0])
  #print(out_endpoints[0].shape, " is shape")
  #print(type(out_endpoints[0]), " is type")
  #array=endpoints['LastBeforeLogits'].eval(session=tf.Session())
  #print(type(array), " is type")
  
  print('\nTop classes and probabilities')
  """
  #for index in sorted_indices[:20]:
    #print(predictions[index], final_logits[index], kinetics_classes[index])


if __name__ == '__main__':
  app.run(main)
