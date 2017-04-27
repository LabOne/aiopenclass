# -*- coding: utf-8 -*-
import tensorflow as tf
#declare some variables with a known value
x = tf.Variable(tf.constant(5.0))
y = tf.Variable(tf.constant(3.0))
#declare variable with random value
z = tf.Variable(tf.random_normal([100], mean=1.0, stddev=2.0))

# perform basic tensorflow math ops
a = x + y
b = a * z
m, v = tf.nn.moments(b, [0])

#initialize tensorflow session, graph, and variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#get the resulting values from the graph nodes
x_val=sess.run(x)
print ('x:',x_val)
z_val=sess.run(z)
print ('z:',z_val)
a_val=sess.run(a)
print ('a:',a_val)
m_val=sess.run(m)
print ('m:',m_val)
v_val=sess.run(v)
print ('v:',v_val)
