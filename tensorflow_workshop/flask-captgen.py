# -*- coding: utf-8 -*-
from flask import Flask, jsonify, render_template, request
import numpy as np
import tensorflow as tf
from capt_gen import Caption_Generator,test
import sys
app = Flask(__name__)

#TODO declare constants for caption generator use
dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 1
learning_rate = 0.001
momentum = 0.9
n_epochs = 25
ixtoword = np.load('data/ixtoword.npy').tolist()
n_words = len(ixtoword)
maxlen=15
captgen=Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words, from_image=True)
print ('initialized')
captgen.build_generator(maxlen=maxlen,from_image=False)
print ('gen built')
saver=tf.train.Saver()
model_path = './models/tensorflow'
saved_path=tf.train.latest_checkpoint(model_path)
saver.restore(captgen.sess, saved_path)
print('restored')
feature_name=sys.argv[1]

def getcaption():
    caption=captgen.get_caption(feature_name)
    return caption

@app.route('/')
def main():
    print ('doing shit')
    return """
    <!doctype html>
    <h1>%s</h1> 
    """ % getcaption()

if __name__=='__main__':
    app.run(host='0.0.0.0',port='8888',debug=True)