import tensorflow as tf
import matplotlib.pyplot as plt

#tf.set_random_seed(777)
# Y = wX+b
x_train = [1, 2, 3]
y_train = [2, 4, 6]

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
'''
GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
==
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
'''

with tf.Session() as sess:

    x_arr = []
    y_arr = []
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        #_, cost_val, W_val, b_val = sess.run([train, cost, w, b])
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: x_train, Y: y_train})
        x_arr.append(step)
        y_arr.append(W_val*step+b_val)

        if step % 100 == 0:
            print(step, cost_val, W_val, b_val)

    plt.plot(x_train, y_train)
    plt.show()
    plt.plot(x_arr, y_arr)
    plt.show()

    print("W : {}, b : {}".format(W_val, b_val))
    print("Y = {}X+{}".format(W_val, b_val))
    print(10 * W_val + b_val)  # [10.000003] / [10.000255] / [10.]
    print(sess.run(hypothesis, feed_dict={X: [1.5, 2.5, 5]}))
    print(sess.run(hypothesis, feed_dict={X: [10]}))
    '''
    0 7.4818788 [0.99133146] [0.33469796]
    100 9.32858e-05 [0.98905206] [0.02488737]
    200 7.182957e-07 [0.99903935] [0.0021838]
    300 5.5318545e-09 [0.9999157] [0.00019162]
    400 4.3205734e-11 [0.9999926] [1.6866776e-05]
    500 4.026409e-13 [0.99999934] [1.5205657e-06]
    600 4.7369517e-15 [0.99999994] [1.2979125e-07]
    700 0.0 [1.] [5.826567e-08]
    800 0.0 [1.] [5.826567e-08]
    900 0.0 [1.] [5.826567e-08]
    1000 0.0 [1.] [5.826567e-08]
    W : [1.], b : [5.826567e-08]
    Y = [1.]X+[5.826567e-08]
    [10.]
    [1.5 2.5 5. ]
    [10.]
    '''
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    for step in range(1001):
        _, cost_val, hypothesis_val = sess.run([train, cost, hypothesis], feed_dict={X: [1, 2, 3, 4, 5], Y: [2, 5, 8, 11, 14]})

        if step % 100 == 0:
            print(step, cost_val, hypothesis_val)

    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

    '''
    0 33.0 [1.38] [0.10000005]
    100 0.19258705 [2.7160506] [0.02514745]
    200 0.097827986 [2.797624] [-0.26935846]
    300 0.049693495 [2.855763] [-0.47925863]
    400 0.025242638 [2.8971996] [-0.6288581]
    500 0.01282239 [2.9267323] [-0.7354806]
    600 0.006513378 [2.9477806] [-0.811472]
    700 0.003308601 [2.9627824] [-0.8656325]
    800 0.0016806467 [2.9734743] [-0.9042339]
    900 0.00085371436 [2.9810948] [-0.93174577]
    1000 0.00043366168 [2.9865258] [-0.95135415]
    [13.981275]
    [6.5149603]
    [3.5284348 9.501486 ]
    '''

