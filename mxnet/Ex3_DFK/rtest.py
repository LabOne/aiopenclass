import random 
import matplotlib.pyplot as plt
import cv2  
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='set up the model path and iteration')
    parser.add_argument('--prefix', type=str, default='.',help='model save path ')
    parser.add_argument('--iteration', type=int, default=50,help='number of iterations ')
    parser.add_argument('--test', type=str, default='.',help='Test data  ')
    
    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)

args=get_args()

def to4d(img):
    return img.reshape(img.shape[0], 1, 96, 96).astype(np.float32) 

model_loaded = mx.model.FeedForward.load(args.prefix, args.iteration)  

X_test = np.load(args.test+'/valset.npz') 
pnum=9 
index=[random.randint(0,len(X_test)) for i in range(pnum)]
print(index)
yvalPred=model_loaded.predict(to4d(X_test[index]))


def plot_sample2(x, y,yval, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10,color='blue')
    axis.scatter(yval[0::2] * 48 + 48, yval[1::2] * 48 + 48, marker='x', s=10,color='red')

fig = pyplot.figure(figsize=(4, 4))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(pnum):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_sample2(X_test[index[i]], y_test[index[i]],yvalPred[i], ax)