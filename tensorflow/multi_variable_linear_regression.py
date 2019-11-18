import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)
# Y = W1X1+W2X2+W3X3+...+WnXn+b

x1_data = [[73., 80., 75., 90., 110.],
          [93., 88., 93., 85., 99.],
          [89., 91., 90., 89., 110.],
          [96., 98., 100., 110., 130.],
          [76., 88., 99., 100., 140.],
          [73., 66., 70., 88., 90]]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]

y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [196.],
          [142.]]
# y_data = [152., 185., 180., 196., 142.]

W1 = tf.Variable(tf.random_normal([5, 1]), name="weight")
# W2 = tf.Variable(tf.random_normal([1]), name="weight2")
# W3 = tf.Variable(tf.random_normal([1]), name="weight3")
b = tf.Variable(tf.random_normal([1]), name="bias")

X1 = tf.placeholder(tf.float32, shape=[None, 5])  # shape=[1, 5]
# X2 = tf.placeholder(tf.float32)
# X3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32, shape=[None, 1])

# hypothesis = X1 * W1 + X2 * W2 + X3 * W3 + b
hypothesis = tf.matmul(X1, W1) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    x_arr = []
    y_arr = []
    sess.run(tf.global_variables_initializer())
    # fd_data = {X1: x1_data, X2: x2_data, X3: x3_data, Y: y_data}
    fd_data = {X1: x1_data, Y: y_data}

    for step in range(40001):
        cost_val, hy_val, w_val, b_bal, _ = sess.run([cost, hypothesis, W1, b, train], feed_dict=fd_data)
        if step % 10000 == 0:
            print(step, "Cost: ", cost_val, "\nw_val: \n", w_val, "\nb_bal: ", b_bal, "\nPrediction:\n", hy_val)

    print(sess.run(hypothesis, feed_dict={X1: [[96., 98., 55., 110., 60.]]}))