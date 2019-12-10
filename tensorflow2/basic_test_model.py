import tensorflow as tf
import numpy as np
import pandas as pd

print(tf.__version__)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))


data = np.random.random((1000, 6))
print(data)
labels = np.random.random((1000, 6))

model_layers = []
layers1 = [tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(6,))]
layers2 = [tf.keras.layers.Dense(36, activation=tf.nn.relu) for i in range(3)]
layers3 = [tf.keras.layers.Dense(24, activation=tf.nn.relu) for i in range(4)]
layers4 = [tf.keras.layers.Dense(6, activation=tf.nn.sigmoid)]



model_layers += layers1 + layers2 + layers3 + layers4



'''
# print(model_layers)

[<tensorflow.python.keras.layers.core.Dense object at 0x1191c3860>, <tensorflow.python.keras.layers.core.Dense object at 0x642b8aef0>, <tensorflow.python.keras.layers.core.Dense object at 0x642ba8240>, <tensorflow.python.keras.layers.core.Dense object at 0x642ba8550>
, <tensorflow.python.keras.layers.core.Dense object at 0x642ba8860>, <tensorflow.python.keras.layers.core.Dense object at 0x642ba8b70>, <tensorflow.python.keras.layers.core.Dense object at 0x642ba8e80>]

'''

keras_model = tf.keras.Sequential(model_layers)

keras_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print(keras_model.summary())

keras_model.fit(data, labels, epochs=10, batch_size=32)


data = np.random.random((1000, 6))
labels = np.random.random((1000, 6))

keras_model.evaluate(data, labels, batch_size=32)

result = keras_model.predict(data, batch_size=32)
print(result.shape)

# perceptron = tf.keras.Sequential(layers)
#
# trunk = tf.keras.Sequential(layers)
# head1 = tf.keras.Sequential(layers1)
# head2 = tf.keras.Sequential(layers2)
#
# path1 = tf.keras.Sequential([trunk, head1])
# path2 = tf.keras.Sequential([trunk, head2])
#
# main_dataset = [[1,2,3,], [4,5,6]]
#
#
# for x, y in main_dataset:
#     with tf.GradientTape() as tape:
#         prediction = path1(x)
#         loss = loss_fn_head1(prediction, y)
#     # trunk와 head1 가중치를 동시에 최적화합니다.
#     gradients = tape.gradient(loss, path1.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, path1.trainable_variables))
