# -*- coding: utf-8 -*-
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import cv2
import skimage

import tensorflow.python.platform
from tensorflow.python.ops import rnn
from keras.preprocessing import sequence
from collections import Counter
import sys
test_image_path='./data/acoustic-guitar-player.jpg'
vgg_path='./data/vgg16-20160129.tfmodel'
class Caption_Generator():
    def __init__(self, dim_in, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, init_b=None,from_image=False):

        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words
        
        if from_image: 
            #load tensorflow model graph for VGG16
            with open(vgg_path,'rb') as f:
                fileContent = f.read()
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fileContent)
            self.images = tf.placeholder("float32", [1, 224, 224, 3])
            tf.import_graph_def(graph_def, input_map={"images":self.images})
            graph = tf.get_default_graph()
            self.sess = tf.InteractiveSession(graph=graph)
            self.graph=graph

        self.from_image=from_image


    def crop_image(self,x, target_height=227, target_width=227, as_float=True,from_path=True):
        #resize image
        image = (x)
        if from_path==True:
            image=cv2.imread(image)
        if as_float:
            image = image.astype(np.float32)

        if len(image.shape) == 2:
            image = np.tile(image[:,:,None], 3)
        elif len(image.shape) == 4:
            image = image[:,:,:,0]

        height, width, rgb = image.shape
        if width == height:
            resized_image = cv2.resize(image, (target_height,target_width))

        elif height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

        return cv2.resize(resized_image, (target_height, target_width))

    def read_image(self,path=None):
        #load and preprocess image
        if path is None:
            path=test_image_path
        img = self.crop_image(path, target_height=224, target_width=224)
        if img.shape[2] == 4:
            img = img[:,:,:3]

        img = img[None, ...]
        return img

    def get_feats(self,x=None):
        #get image featurization from vgg16
        feat=self.read_image(x)
        fc7 = self.sess.run(self.graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={self.images:feat})
        return fc7

dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 1
momentum = 0.9
n_epochs = 25

if __name__=='__main__':
    in_name=sys.argv[1]
    out_name=sys.argv[2]
    ixtoword = np.load('data/ixtoword.npy').tolist()
    n_words = len(ixtoword)
    maxlen=15
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words, np.zeros(dim_embed).astype(np.float32),from_image=True)
    feats=caption_generator.get_feats(in_name)
    feats=feats.reshape([1,dim_embed])
    np.save(out_name,feats)
