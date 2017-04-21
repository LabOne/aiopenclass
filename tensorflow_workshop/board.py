import tensorflow as tf

# pull MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# construction phase
x = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder for input data (images)
y = tf.placeholder(tf.float32, shape=[None, 10]) # placeholder for label data

with tf.name_scope('fc_1'): # first fully connected layer
    W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
    h = tf.sigmoid(tf.matmul(x, W1) + b1)
    tf.summary.histogram('layer1_weights', W1) 

with tf.name_scope('fc_2'): # second fully connected layer
    W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    y_predict = tf.matmul(h, W2) + b2

with tf.name_scope('eval'): 
    with tf.name_scope('loss'): # calculating loss for the neural network
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))
        tf.summary.scalar('loss', cross_entropy)
    backprop = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # optimizer backpropagation step

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# execution phase
sess = tf.Session()
merged = tf.summary.merge_all() # compile all summaries
writer = tf.summary.FileWriter("graphs", sess.graph) # writer for events file (graph and learning visualization)
sess.run(tf.global_variables_initializer()) # variable initialization step

train_steps = 20000
batch_size = 50
for i in range(train_steps):
    batch_x, batch_y = data.train.next_batch(batch_size) # collect next batch of input data and labels
    if i % 10 == 0:
        summary, _ = sess.run([merged, backprop], feed_dict={x: batch_x, y: batch_y})
        writer.add_summary(summary, i) # write summaries every 10 training steps
    else:
        sess.run(backprop, feed_dict={x: batch_x, y: batch_y})

# testing accuracy of trained neural network
print(sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels}))
