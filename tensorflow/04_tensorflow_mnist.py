import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)


print("훈련 이미지 :",  mnist.train.images.shape)
print("훈련 라벨:",  mnist.train.labels.shape)
print("테스트 이미지 : ", mnist.test.images.shape)
print("테스트 라벨 : ", mnist.test.labels.shape)
print("검증 이미지 : ", mnist.validation.images.shape)
print("검증 라벨 : ", mnist.validation.labels.shape)
print('\n')

mnist_idx = 200

print('[label]')
print('one-hot vector label = ', mnist.train.labels[mnist_idx])
print('number label = ', np.argmax(mnist.train.labels[mnist_idx]))
print('\n')
#
# print('[image]')
#
# for index, pixel in enumerate(mnist.train.images[mnist_idx]):
#     if index % 28 == 0:
#         print('\n')
#     else:
#         print("%10f" % pixel, end="")
# print('\n')

# plt.figure(figsize=(5, 5))
# image = np.reshape(mnist.train.images[mnist_idx], [28, 28])
# plt.imshow(image, cmap='Greys')
# plt.imshow(image)
# plt.show()


nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# 10개의 값 중 가장 확률이 높은 것을 고르기 위해 Softmax 사용
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 50
batch_size = 500
num_iterations = int(mnist.train.num_examples / batch_size)


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0
        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )

    correct_vals = sess.run(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print ('전체 테스트 데이터', len(correct_vals), '중에 정답수:', len(correct_vals[correct_vals == True]), ', 오답수:', len(correct_vals[correct_vals == False]))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    print("\n\n===================검증 데이터를 통한 결과 확인===========================================\n\n")

    fig = plt.figure(figsize=(25, 5))
    for i in range(10):
        n = random.randint(0, mnist.validation.num_examples - 1)
        # validation_label = sess.run(tf.argmax(mnist.validation.labels[n : n + 1], 1))
        validation_label = np.argmax(mnist.validation.labels[n])
        test_prediction = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.validation.images[n: n+1]})

        print("label : {} , prediction : {}".format(validation_label, test_prediction))

        ax = fig.add_subplot(1, 10, i+1)
        im = np.reshape(mnist.validation.images[n], [28, 28])
        ax.imshow(im, cmap='Greys')
        ax.text(0, -2, 'validation_label=' + str(validation_label) + '\ntest_prediction=' + str(test_prediction))

        # plt.figure(figsize=(5, 5))
        # plt.imshow(
        #     mnist.validation.images[n : n + 1].reshape(28, 28),
        #     cmap="Greys",
        #     interpolation="nearest"
        # )
    plt.show()
    sess.close()

''' learning_rate = 0.01 & batch_size = 100
Epoch: 0001, Cost: 9.648579367
Epoch: 0010, Cost: 1.348176614
Epoch: 0020, Cost: 0.948549482
Epoch: 0030, Cost: 0.799144279
Epoch: 0040, Cost: 0.714591498
Epoch: 0050, Cost: 0.658038826
Learning finished
Accuracy:  0.8589
Label:  [2]
Prediction:  [2]
'''

''' learning_rate = 0.1 & batch_size = 100
Epoch: 0001, Cost: 2.689707846
Epoch: 0010, Cost: 0.545459472
Epoch: 0020, Cost: 0.434066005
Epoch: 0030, Cost: 0.384364012
Epoch: 0040, Cost: 0.354346280
Epoch: 0050, Cost: 0.333534072
Learning finished
Accuracy:  0.9078
Label:  [3]
Prediction:  [3]
'''

''' learning_rate = 0.5 & batch_size = 100
Epoch: 0001, Cost: 1.275443188
Epoch: 0010, Cost: 0.346373933
Epoch: 0020, Cost: 0.296955085
Epoch: 0030, Cost: 0.276751447
Epoch: 0040, Cost: 0.266433251
Epoch: 0050, Cost: 0.258470672
Learning finished
Accuracy:  0.9214
Label:  [8]
Prediction:  [8]
'''

''' validation check
label : [5] , prediction : [5]
label : [7] , prediction : [7]
label : [7] , prediction : [7]
label : [6] , prediction : [6]
label : [2] , prediction : [2]
label : [8] , prediction : [8]
label : [8] , prediction : [8]
label : [9] , prediction : [9]
label : [9] , prediction : [9]
label : [9] , prediction : [9]
'''