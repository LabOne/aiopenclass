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
    parser.add_argument('--TEST', type=int, default=900,
                        help='TEST size')
   
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

#input parameter which decides the class & the rest for binary classication
#m=int(sys.argv[1])
m = 0

X = np.empty([TRAIN*CLASS, BSIZE*HSIZE], dtype=np.float32)
y = np.empty(TRAIN*CLASS, dtype=int32)
y_pred = np.empty(TRAIN*CLASS, dtype=int32)

Xval = np.empty([TEST*CLASS, 160*120], dtype=np.float32)
yval = np.empty(TEST*CLASS, dtype=int32)

i=0
j=0
jval=0
# This would print all the files and directories
basicPath=args.input
for i in range(0,10):
    path = basicPath+"/imgs/train/c" + str(i) + "/"
    dirs = os.listdir( path )
    k=0
    for file in dirs:
        if k>(TRAIN+TEST)-1: break
        imagefile = path + file
        img = Image.open(imagefile)
        img = img.resize((BSIZE,HSIZE), Image.ANTIALIAS).convert('L')
        img_arr = np.asarray(img.getdata(),dtype=np.float64)
        img_arr = np.asarray(img_arr,dtype=np.uint8) 
        if k<TRAIN: 
            X[j,] = img_arr.ravel() / 255.  # scale pixel values to [0, 1]
            y[j] = i
            j+=1
        else:
            Xval[jval,] = img_arr.ravel() / 255.  # scale pixel values to [0, 1]
            yval[jval] = i
            jval+=1
        k+=1

X = X.reshape(-1, 1, BSIZE, HSIZE)
Xval = Xval.reshape(-1, 1, 160, 120)

# c = np.fromfile('test2.dat', dtype=int)

print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

print("Xval.shape == {}; Xval.min == {:.3f}; Xval.max == {:.3f}".format(
    Xval.shape, Xval.min(), Xval.max()))
print("yval.shape == {}; yval.min == {:.3f}; yval.max == {:.3f}".format(
    yval.shape, yval.min(), y.max()))
print(stats.itemfreq(y))
print(stats.itemfreq(yval))

np.savez(args.output+'/trainset.npz',X=X,y=y)
np.savez(args.output+'/valset.npz',Xval=Xval,yval=yval)
#np.savezX.tofile(args.output+'/TrainX')
#Xval.tofile(args.output+'/ValX')
#y.tofile(args.output+'/TrainY')
#yval.tofile(args.output+'/ValY')

