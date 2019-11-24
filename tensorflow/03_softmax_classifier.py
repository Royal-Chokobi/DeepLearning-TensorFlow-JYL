import tensorflow as tf
# tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])
nb_classes = 3

# Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
# print("one_hot:", Y_one_hot)
# Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# print("reshape one_hot:", Y_one_hot)
'''
one_hot: Tensor("one_hot:0", shape=(?, 3, 3), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 3), dtype=float32)
'''


W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

            if step % 200 == 0:
                print(step, cost_val)

    print('--------------')
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))


'''

0 6.926112
200 0.6005017
400 0.47295794
600 0.37342948
800 0.28018376
1000 0.23280525
1200 0.21065348
1400 0.19229904
1600 0.17682332
1800 0.16359553
2000 0.15216157
--------------
[[1.3890439e-03 9.9860197e-01 9.0613530e-06]] [1]
--------------
[[0.9311919  0.06290217 0.00590592]] [0]
--------------
[[1.2732816e-08 3.3411355e-04 9.9966586e-01]] [2]
--------------
[[1.3890439e-03 9.9860197e-01 9.0613357e-06]
 [9.3119192e-01 6.2902212e-02 5.9059230e-03]
 [1.2732840e-08 3.3411387e-04 9.9966586e-01]] [1 0 2]
 
 '''