import pickle
import scipy.stats as stats  

import mxnet as mx  
from numpy import *
import os, sys
import numpy as np
from scipy.misc import imread, imsave, imresize
from PIL import Image, ImageDraw
import argparse 


def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='set up the input path of image files')
    parser.add_argument('--input', type=str, default='.',
                        help = 'the input path of imgs')
    parser.add_argument('--output', type=str, default='.',
                        help='the content image')
    parser.add_argument('--TRAIN', type=int, default=1000,
                        help='Train size')
    parser.add_argument('--TEST', type=int, default=-1,
                        help='TEst size')
   
    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)

if __name__=='__main__':
    args=get_args()
    print args.input





