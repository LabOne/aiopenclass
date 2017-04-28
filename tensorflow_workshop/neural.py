# -*- coding: utf-8 -*-
import tensorflow as tf

# pull MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_steps = 2000
batch_size = 50
avg_train_loss=0
display_step=100

# construction phase
x = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder for input data (images)
y = tf.placeholder(tf.float32, shape=[None, 10]) # placeholder for label data

# create variables for Wx+b multiplication
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
#perform nonlinearity
h = tf.sigmoid(tf.matmul(x, W1) + b1)

# create variables for W2h+b2 multiplication
W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
#get -log probabilities of predictions (logits) 
y_predict = tf.matmul(h, W2) + b2 #tf.nn.softmax(tf.matmul(h, W2) + b2)

#get classification loss from logits
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))

#Reason for using logits has to do with numerical stability and what to do with zero values

# Declare Global Step (counter)
global_step=tf.Variable(0,trainable=False)
# Variable learning rate
learning_rate=0.5
learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           display_step, 0.95, staircase=True)

backprop = tf.train.GradientDescentOptimizer(learning_rate)
########################## 
# We break apart the following call to minimize into it's 4 parts 
# backprop = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # optimizer backpropagation step
# 1.getting trainable variables
# 2.computing gradients
# 3.clipping or quantizing gradients
# 4.applying gradients 
##########################

tvars=tf.trainable_variables()
grads,_=tf.clip_by_global_norm(tf.gradients(cross_entropy,tvars),.1)

backprop= backprop.apply_gradients(zip(grads,tvars))

correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# execution phase
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # variable initialization step

for i in range(train_steps):
    batch_x, batch_y = data.train.next_batch(batch_size) # collect next batch of input data and labels
    #perform gradient update and get loss
    _,loss=sess.run([backprop,cross_entropy], feed_dict={x: batch_x, y: batch_y})
    avg_train_loss=avg_train_loss*i/(i+1) + loss/(i+1)
    if i%display_step==0:
	    print ('average loss:',avg_train_loss)

# testing accuracy of trained neural network
print(sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels}))
