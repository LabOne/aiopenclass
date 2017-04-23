import numpy as np
from scipy.misc import imread, imsave, imresize
from PIL import Image, ImageDraw
import argparse 
import mxnet as mx 

def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='set up the parameters for evaluating the NN')
    parser.add_argument('--prefix', type=str, default='fullymodel',
                        help = 'the model file name')
    parser.add_argument('--iteration', type=int, default=10,
                        help = 'the model file name')
    parser.add_argument('--output', type=str, default='.',
                        help='the output path')
    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)

args=get_args()

prefix =args.prefix 
model_loaded = mx.model.FeedForward.load(prefix, iteration)   
valdata = np.load(args.output+'/valset.npz')
Xval=traindata['Xval']
yval=data['yval']
 
yval_p=model_loaded.predict(Xval) 

ind=np.argsort(yval_p)
tp,fp=0,0
tn,fn=0,0
mlogloss=0 
for i in range(len(yval)):
    mlogloss
    if yval[i]==0:
        if yval_p[i].argmax()==0: tp+=1 
        else: fn+=1     
    else:     
        if yval_p[i].argmax()!=0: tn+=1
        else:  fp+=1 
    mlogloss+=-math.log(yval_p[i][yval[i]])   
mlogloss=mlogloss/len(yval)                
pd=tp/float(tp+fn)
accuracy=float(tp+tn)/float(tp+fp+tn+fn)
precision0=float(tp)/float(tp+fp)
precision1=float(tn)/float(tn+fn)
print('pd=',pd,'; accuracy=',accuracy,'; precision0',
           precision0,'; precision1:',precision1,'; mlogloss:',mlogloss)
        
