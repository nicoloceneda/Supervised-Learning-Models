import tensorflow as tf
import time

start = time.time()

# Create a graph

g = tf.Graph()

with g.as_default():

    x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')

    z = tf.add(tf.multiply(w, x), b, name='z')

    init = tf.global_variables_initializer()

# Create a session and pass in graph g

with tf.Session(graph=g) as sess:

    sess.run(init)

    for t in [1.0, 0.6, -1.8]:

        print('x = {:4} --> z = {:5.2f}'.format(t, sess.run(z, feed_dict={x: t})))

end = time.time()

print('Execution time: ', end-start)