# Test the results using the pretrained model 
import random 
import numpy as np 
import cv2 
from os import listdir

import matplotlib 
import matplotlib.pyplot as pyplot 
from PIL import Image, ImageDraw 
import mxnet as mx 

prefix = "fullymodel" # Trained using AWS GPU 
fullymodel_loaded = mx.model.FeedForward.load(prefix, 50) 

CLASS=10
BSIZE=160
HSIZE=120
bpath='./imgs/train'
pnum=6 
dictM={0:'safe driving',1:'texting - right',2:'talking on the phone - right',
  3:'texting - left',4:'talking on the phone - left',
  5:'operating the radio',6:'drinking',7:'reaching behind',
  8:'hair and makeup',9:'talking to passenger'}
pallfiles=[]
fig = pyplot.figure(figsize=(12,7))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
cls=range(0,10)
random.shuffle(cls)

for i in range(0,pnum):
    subd=cls[i]
    bpathimg=bpath+'/c'+str(subd)
    files=listdir(bpathimg)
    imgpath=bpathimg+'/'+files[random.randint(0,(len(files)-1))]
    img = Image.open(imgpath)
    img = img.resize((BSIZE,HSIZE), Image.ANTIALIAS).convert('L')
    img_arr = np.asarray(img.getdata(),dtype=np.float64)
    img_arr = np.asarray(img_arr,dtype=np.uint8) 
    XX = np.empty([1, BSIZE*HSIZE], dtype=np.float32)
    XX[0,]= img_arr.ravel() / 255.  # scale pixel values to [0, 1]
    XX = XX.reshape(-1, 1, BSIZE, HSIZE)
    prob=fullymodel_loaded.predict(XX)[0]
    label = dictM[prob.argmax()]
    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[])
    ax.set_title(label) 
    ax.imshow(img) 
pyplot.show() 
