# -*- coding: utf-8 -*-
"""
Based off of the tensorflow distributed training example
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10


import sys
import pickle as pkl
NUM_GPUS=1


def tower_loss(scope,model):
  """Calculate the total loss on a single tower running the model.
  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'
    model: model instantiation of the tower
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  total_loss=model.loss

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train_distributed(network_architecture, object_constructor, model_path,learning_rate=0.001,
      batch_size=100, training_epochs=10, display_step=2,gen=False,ctrain=False,test=False):
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    if should_decay and not gen:
      global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                             int(all_samps/NUM_GPUS), 0.95, staircase=True)
    opt=tf.train.AdamOptimizer(learning_rate)
    tower_grads=[]
    og_batch_size=batch_size
    batch_size=batch_size*NUM_GPUS
    x_placeholders=[]
    y_placeholders=[]
    mask_placeholders=[]
    losses=[]
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(NUM_GPUS):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('model_%d' % (i,)) as scope:
            # TODO include variable restoring
            # Treat this as a black box. This object constructor is for another model, not the caption generator we made 
            vae = object_constructor(network_architecture, 
                           learning_rate=learning_rate, 
                           batch_size=batch_size,generative=gen,ctrain=ctrain,test=test,global_step=global_step)
            loss=distributed_train.tower_loss(scope,vae)
            losses.append(loss)
            #get all the placeholder inputs for each model
            x_placeholders.append(vae.x)
            y_placeholders.append(vae.caption_placeholder)
            mask_placeholders.append(vae.mask)
            # Training cycle variable reuse so same variable is used across name_scopes
            tf.get_variable_scope().reuse_variables()

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
    #average loss across towers
    loss=tf.concat(losses,axis=0)
    loss=tf.reduce_mean(loss)
    #average gradients across towers and apply gradient update to variables
    grads=distributed_train.average_gradients(tower_grads)
    apply_opt=opt.apply_gradients(grads,global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op=tf.group([apply_opt,variables_averages_op])
    #initialize saver, graph/variables, and session
    saver=tf.train.Saver(tf.global_variables())
    init=tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init)
    costs=[]
    indlist=np.arange(all_samps).astype(int)
    total_batch=int(training_epochs/batch_size)
    #iterate over epochs
    for epoch in range(training_epochs):
      np.random.shuffle(indlist)
      avg_loss=0
      #iterate over an epoch
      for i in range(total_batch):
        batch_xs = X[indlist[i*batch_size:(i+1)*batch_size]]
        batch_ys=y[indlist[i*batch_size:(i+1)*batch_size]].astype(np.uint32)
        batch_mask[indlist[i*batch_size:(i+1)*batch_size]]
        
        # loop over models and divide batch data into NUM_GPUS slices and create feed dict appropriately
        # run session and get losses n shit
        feed_dict={}
        for model in range(NUM_GPUS):
          feed_dict[x_placeholders[model]]=batch_xs[model*og_batch_size:(model+1)*og_batch_size]
          feed_dict[y_placeholders[model]]=batch_ys[model*og_batch_size:(model+1)*og_batch_size]
          feed_dict[mask_placeholders[model]]=batch_mask[model*og_batch_size:(model+1)*og_batch_size]
        _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)
        avg_loss=avg_loss*i/(i+1)+loss_value/(i+1)
        if epoch==0 and i==0:
          print ('Epoch: 0', 'cost=', avg_loss)
          costs.append(avg_loss)
      costs.append(avg_loss)
      if epoch %display_step==0 or epoch==1:
        saver.save(sess,os.path.join(model_path,'model'))
        print ("Epoch:",'%04d'%(epoch+1),'cost=',avg_loss)

