import pickle
import scipy.stats as stats  

import mxnet as mx  
from numpy import *
import os, sys
import numpy as np
from scipy.misc import imread, imsave, imresize
from PIL import Image, ImageDraw
import argparse 
import cnndd 


def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='set up the input path of image files')
    parser.add_argument('--input', type=str, default='.',
                        help = 'the input path of imgs')
    parser.add_argument('--output', type=str, default='.',
                        help='the content image')
    parser.add_argument('--TRAIN', type=int, default=1000,
                        help='Train size')
    parser.add_argument('--TEST', type=int, default=900,
                        help='TEST size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning_rate')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='momentum')
    parser.add_argument('--wd', type=float, default=0.001,
                        help='wd')
    parser.add_argument('--num_epoch', type=int, default=10,
                        help='num_epoch')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu')
    parser.add_argument('--mpath', type=str, default='.',
                        help='model save path ')

    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)

args=get_args()
#Separate the data into train and test datasets 

CLASS=10
MAX_EPOCHS=40
TRAIN=args.TRAIN
TEST=args.TEST
BSIZE=160
HSIZE=120

traindata = np.load(args.output+'/trainset.npz')
X=traindata['X']
y=traindata['y']

valdata = np.load(args.output+'/valset.npz')
Xval=valdata['Xval']
yval=valdata['yval']

import logging
logging.getLogger().setLevel(logging.DEBUG)

device = mx.cpu() if args.gpu==-1 else mx.gpu(args.gpu) 
model = mx.model.FeedForward(
    symbol = cnndd.distDrCNN(),       # network structure
    ctx=device,
    num_epoch =args.num_epoch,     # number of data passes for training 
    learning_rate=args.learning_rate,
    momentum = args.momentum,
    wd= 0.001,
    initializer=mx.init.Uniform(0.1),
   #initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34)
)
batch_size=500
model.fit(
    X=X,y=y,       # training data
    eval_data=(Xval,yval), # validation data
    eval_metric='acc',
    batch_end_callback = mx.callback.Speedometer(batch_size, 200)
)

prefix = 'ddrmodel'
iteration = 50
model.save(prefix) 


