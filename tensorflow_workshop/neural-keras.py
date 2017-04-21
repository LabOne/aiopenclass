import tensorflow as tf2
from keras.models import Sequential

# pull MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# construction phase
x = tf.placeholder(tf.float32, shape=[None, 784]) # placeholder for input data (images)
y = tf.placeholder(tf.float32, shape=[None, 10]) # placeholder for label data

model=Sequential()
model.add(Dense(200))
model.add(Activation('sigmoid')) 

#note the activation can be folded into the layer as follow
model.add(Dense(10))

y_predict=model(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))
backprop = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # optimizer backpropagation step



# execution phase
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # variable initialization step

train_steps = 2000
batch_size = 50
for i in range(train_steps):
    batch_x, batch_y = data.train.next_batch(batch_size) # collect next batch of input data and labels
    sess.run(backprop, feed_dict={x: batch_x, y: batch_y})

#declare accuracy operations
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# testing accuracy of trained neural network
print(sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels}))
