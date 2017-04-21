import tensorflow as tf

x = tf.Variable(tf.constant(5.0))
y = tf.Variable(tf.constant(3.0))
z = tf.Variable(tf.random_normal([100], mean=1.0, stddev=2.0))

a = x + y
b = a * z
m, v = tf.nn.moments(b, [0])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(x)
sess.run(z)
sess.run(a)
sess.run(m)
sess.run(v)
