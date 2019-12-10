import tensorflow as tf
import numpy as np

# xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]

# print(x_data.shape, y_data.shape)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    print(sess.run([predicted], feed_dict={X: [[1.5, 2.5]]}))
    print(sess.run([predicted], feed_dict={X: [[3, 1]]}))
    print(sess.run([predicted], feed_dict={X: [[1, 3]]}))
    print(sess.run([predicted], feed_dict={X: [[0, 9]]}))
    print(sess.run([predicted], feed_dict={X: [[6, 1]]}))
    print(sess.run([predicted], feed_dict={X: x_data}))
'''
0 3.0121937
2000 0.35643885
4000 0.26782897
6000 0.21212512
8000 0.17498533
10000 0.14879644

Hypothesis:  [[0.03044671]
 [0.158447  ]
 [0.30346012]
 [0.7820259 ]
 [0.9399801 ]
 [0.9803049 ]] 
Correct (Y):  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
Accuracy:  1.0
[array([[0.]], dtype=float32)]
[array([[0.]], dtype=float32)]
[array([[0.]], dtype=float32)]
[array([[0.]], dtype=float32)]
[array([[1.]], dtype=float32)]
[array([[0.],
       [0.],
       [0.],
       [1.],
       [1.],
       [1.]], dtype=float32)]

'''
