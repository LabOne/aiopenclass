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
            with open(vgg_path,'rb') as f:
                fileContent = f.read()
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fileContent)
            self.images = tf.placeholder("float32", [1, 224, 224, 3])
            tf.import_graph_def(graph_def, input_map={"images":self.images})
            graph = tf.get_default_graph()
            self.sess = tf.InteractiveSession(graph=graph)

        self.from_image=from_image

        # declare the variables to be used for our word embeddings
        with tf.device("/cpu:0"):
            self.word_embedding = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='word_embedding')

        self.embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')
        
        # declare the LSTM itself
        self.lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
        
        # declare the variables to be used to embed the image feature embedding to the word embedding space
        self.img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

        # declare the variables to go from an LSTM output to a word encoding output
        self.word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
        # initialize this bias variable from the preProBuildWordVocab output
        # optional initialization setter for encoding bias variable 
        if init_b is not None:
            self.word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')
        else:
            self.word_encoding_bias = tf.Variable(tf.zeros([n_words]), name='word_encoding_bias')

    def build_model(self):
        # declaring the placeholders for our extracted image feature vectors, our caption, and our mask
        # (describes how long our caption is with an array of 0/1 values of length `maxlen`  
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        caption_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])
        
        # getting an initial LSTM embedding from our image_imbedding
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        
        #flatten timeseries to get all word embeddings at once
        flat_caption_placeholder=tf.reshape(caption_placeholder,[-1,1])
        #leverage one-hot sparsity to lookup embeddings fast
        with tf.device('/cpu:0'):
            word_embeddings=tf.nn.embedding_lookup(self.word_embedding,flat_caption_placeholder)
        word_embeddings+=self.embedding_bias
        word_embeddings=tf.reshape(word_embeddings,[self.batch_size,self.n_lstm_steps,-1])
        #initialize lstm state
        state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
        rnn_output=[]
        with tf.variable_scope("RNN"):
            # unroll lstm
            for i in range(self.n_lstm_steps): 
                if i > 0:
                   # if this isn’t the first iteration of our LSTM we need to get the word_embedding corresponding
                   # to the (i-1)th word in our caption 
                    
                    current_embedding = word_embeddings[:,i-1,:]
                else:
                     #if this is the first iteration of our LSTM we utilize the embedded image as our input 
                    current_embedding = image_embedding
                if i > 0: 
                    # allows us to reuse the LSTM tensor variable on each iteration
                    tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(current_embedding, state)

                rnn_output.append(tf.expand_dims(out,1))
        #perform classification of output
        rnn_output=tf.concat(rnn_output,axis=1)
        rnn_output=tf.reshape(rnn_output,[self.batch_size*self.n_lstm_steps,-1])
        encoded_output=tf.matmul(rnn_output,self.word_encoding)+self.word_encoding_bias
        #get loss
        xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=encoded_output,labels=tf.reshape(caption_placeholder,[-1]))
        #mask zero embeddings
        masked_xentropy=tf.multiply(tf.reshape(xentropy,[self.batch_size,-1])[:,1:],mask[:,1:])
        #average over timeseries length
        total_loss=tf.reduce_sum(masked_xentropy)/tf.reduce_sum(mask[:,1:])
        return total_loss, img,  caption_placeholder, mask

    def build_generator(self, maxlen, batchsize=1,from_image=False):
        #same setup as `build_model` function

        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(batchsize,dtype=tf.float32)

        #declare list to hold the words of our generated captions
        all_words = []
        with tf.variable_scope("RNN"):
            # in the first iteration we have no previous word, so we directly pass in the image embedding
            # and set the `previous_word` to the embedding of the start token ([0]) for the future iterations
            output, state = self.lstm(image_embedding, state)
            previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.embedding_bias

            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(previous_word, state)


                # get a get maximum probability word and it's encoding from the output of the LSTM
                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)

                with tf.device("/cpu:0"):
                    # get the embedding of the best_word to use as input to the next iteration of our LSTM 
                    previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

                previous_word += self.embedding_bias

                all_words.append(best_word)
        self.img=img
        self.all_words=all_words
        return img, all_words
    def crop_image(self,x, target_height=227, target_width=227, as_float=True,from_path=True):
        #image preprocessing to crop and resize image
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
        # parses image from file path and crops/resizes
        if path is None:
            path=test_image_path
        img = crop_image(path, target_height=224, target_width=224)
        if img.shape[2] == 4:
            img = img[:,:,:3]

        img = img[None, ...]
        return img

    def get_caption(self,x=None):
        #gets caption from an image by feeding it through imported VGG16 graph
        if self.from_image:
            feat = read_image(x)
            fc7 = self.sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={self.images:feat})
        else:
            fc7=np.load(x,'r')
        
        generated_word_index= self.sess.run(self.generated_words, feed_dict={self.img:fc7})
        generated_word_index = np.hstack(generated_word_index)
        generated_words = [ixtoword[x] for x in generated_word_index]
        punctuation = np.argmax(np.array(generated_words) == '.')+1

        generated_words = generated_words[:punctuation]
        generated_sentence = ' '.join(generated_words)
        return (generated_sentence)
def get_data(annotation_path, feature_path):
    #load training/validation data
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    return np.load(feature_path,'r'), annotations['caption'].values
def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # function from Andre Karpathy's NeuralTalk
    #process and vectorize training/validation captions
    print('preprocessing %d word vocab' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  
    wordtoix = {}
    wordtoix['#START#'] = 0 
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) 
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) 
    return wordtoix, ixtoword, bias_init_vector.astype(np.float32)

dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 128
momentum = 0.9
n_epochs = 25

def train(learning_rate=0.001, continue_training=False):
    
    tf.reset_default_graph()

    feats, captions = get_data(annotation_path, feature_path)
    wordtoix, ixtoword, init_b = preProBuildWordVocab(captions)

    np.save('data/ixtoword', ixtoword)
    print ('num words:',len(ixtoword))

    


    sess = tf.InteractiveSession()
    n_words = len(wordtoix)
    maxlen = np.max( [x for x in map(lambda x: len(x.split(' ')), captions) ] )
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words, init_b)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=100)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.global_variables_initializer().run()

    if continue_training:
        saver.restore(sess,tf.train.latest_checkpoint(model_path))

    for epoch in range(n_epochs):
        index = (np.arange(len(feats)).astype(int))
        np.random.shuffle(index)
        index=index[:(batch_size*10)]
        for start, end in zip( range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):
            #format data batch
            current_feats = feats[index[start:end]]
            current_captions = captions[index[start:end]]
            current_caption_ind = [x for x in map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)]

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] )

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([x for x in map(lambda x: (x != 0).sum()+2, current_caption_matrix )])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats.astype(np.float32),
                sentence : current_caption_matrix.astype(np.int32),
                mask : current_mask_matrix.astype(np.float32)
                })

            print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs), "\t Iter {}/{}".format(start,len(feats)))

        print("Saving the model from epoch: ", epoch)
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        learning_rate *= 0.95
def test(sess,image,generated_words,ixtoword,idx=0): # Naive greedy search

    feats, captions = get_data(annotation_path, feature_path)
    feat = np.array([feats[idx]])
    
    saver = tf.train.Saver()
    sanity_check= False
    # sanity_check=True
    if not sanity_check:
        saved_path=tf.train.latest_checkpoint(model_path)
        saver.restore(sess, saved_path)
    else:
        tf.global_variables_initializer().run()

    generated_word_index= sess.run(generated_words, feed_dict={image:feat})
    generated_word_index = np.hstack(generated_word_index)

    generated_sentence = [ixtoword[x] for x in generated_word_index]
    print(generated_sentence)

if __name__=='__main__':

    model_path = './models/tensorflow'
    feature_path = './data/feats.npy'
    annotation_path = './data/results_20130124.token'
    import sys
    feats, captions = get_data(annotation_path, feature_path)
    if sys.argv[1]=='train':
        train()
    elif sys.argv[1]=='test':
        ixtoword = np.load('data/ixtoword.npy').tolist()
        n_words = len(ixtoword)
        maxlen=15
        sess = tf.InteractiveSession()
        batch_size=1
        caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen+2, n_words)


        image, generated_words = caption_generator.build_generator(maxlen=maxlen)
        test(sess,image,generated_words,ixtoword,1)