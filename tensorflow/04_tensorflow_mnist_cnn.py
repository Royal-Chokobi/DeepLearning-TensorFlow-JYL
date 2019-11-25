import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

""" 가중치 초기화 """
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

""" Bias 초기화 """
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Convolution 정의 """
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

""" Pooling 정의 """
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

""" placeholder 정의 : 데이터가 들어 갈 곳
이미지와 정답 레이블용 2차원 tensor를 만든다.
None은 어떤 length도 가능함을 의미한다. """
# 이미지 데이터용 placeholder
x = tf.placeholder(tf.float32, [None, 784])
# 정답 레이블용 placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

""" Variable 정의 : 학습 결과가 저장될 가중치(weight)와 바이어스(bias) """
# 0으로 초기화 함
W = tf.Variable(tf.zeros([784, 10])) # w는 784차원의 이미지 벡터를 곱해, 10차원(one hot encoding된 0~9)의 결과를 내기 위한 것
b = tf.Variable(tf.zeros([10]))      # b는 결과에 더해야 하므로 10차원

""" 모델 정의 : Softmax Regression
10개의 값 중 가장 확률이 높은 것을 고르기 위해 Softmax 사용 """
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 입력 데이터를 4D 텐서로 재정의
# 두 번째/세 번째 파라미터는 이미지의 가로/세로 길이
# 마지막 파라미터 컬러 채널의 수는 흑백 이미지이므로 1임
x_image = tf.reshape(x, [-1,28,28,1])

""" First Convolutional Layer 정의 """
# 가중치 텐서 정의(patch size, patch size, input channel, output channel).
# 5x5의 윈도우(patch라고도 함) 크기를 가지는 32개의 feature(kernel, filter)를 사용
# 흑백 이미지이므로 input channel은 1임
W_conv1 = weight_variable([5, 5, 1, 32])
# 바이어스 텐서 정의
b_conv1 = bias_variable([32])
# x_image와 가중치 텐서에 합성곱을 적용하고, 바이어스을 더한 뒤 ReLU 함수를 적용
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 출력값을 구하기 위해 맥스 풀링을 적용
h_pool1 = max_pool_2x2(h_conv1)

""" Second Convolutional Layer 정의 """
# 가중치 텐서 정의(patch size, patch size, input channel, output channel)
# 5x5의 윈도우(patch라고도 함) 크기를 가지는 64개의 feature를 사용
# 이전 레이어의 output channel의 크기가 32가 여기에서는 input channel이 됨
W_conv2 = weight_variable([5, 5, 32, 64])
# 바이어스 텐서 정의
b_conv2 = bias_variable([64])
# First Convolutional Layer의 출력값인 h_pool1과 가중치 텐서에 합성곱을 적용하고, 바이어스을 더한 뒤 ReLU 함수를 적용
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 출력값을 구하기 위해 맥스 풀링을 적용
h_pool2 = max_pool_2x2(h_conv2)

""" 완전 연결 레이어(Fully-Connected Layer) 정의 """
#  7×7 크기의 64개 필터. 임의로 선택한 뉴런의 갯수(여기서는 1024)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

""" Dropout 정의 """
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

""" 최종 소프트맥스 계층 정의 """
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 모델 훈련 및 평가
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1001):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

'''
step 0, training accuracy 0.13
step 100, training accuracy 0.81
step 200, training accuracy 0.93
step 300, training accuracy 0.92
step 400, training accuracy 0.97
step 500, training accuracy 0.96
step 600, training accuracy 0.93
step 700, training accuracy 1
step 800, training accuracy 0.97
step 900, training accuracy 0.95
step 1000, training accuracy 0.94
test accuracy 0.9681
'''