import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(tf.__version__)
tf.keras.backend.clear_session()


def create_time_steps(length):
    return list(range(-length, 0))


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


file_dir = os.path.dirname(os.path.realpath(__file__))
finance_data = pd.read_csv(file_dir+'/finance.csv')
print(finance_data.dtypes)
print(finance_data.head())

uni_data = finance_data['Adj_Close']
uni_data.index = finance_data['realDate']
uni_data.plot(subplots=True)
plt.show()

wonUpDown_data = finance_data['Adj_Won']
wonUpDown_data.index = finance_data['realDate']
wonUpDown_data.plot(subplots=True)
plt.show()

fin_real_date = finance_data.pop('realDate')

data_len = len(finance_data)
fn_date = finance_data.pop('Date')

finance_df = finance_data
finance_df.index = fin_real_date
print(finance_df.head())

print("="*200)

BATCH_SIZE = 100
BUFFER_SIZE = 1000
EVALUATION_INTERVAL = 200
EPOCHS = 1

finance_df.plot(subplots=True)
plt.show()

TRAIN_SPLIT = 3000
dataset = finance_df.values
data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)
dataset = (dataset-data_mean)/data_std

print(dataset)
print("="*200)

history_size = 20
target_size = 1
STEP = 1

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, history_size,
                                                 target_size, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, history_size,
                                             target_size, STEP)


print('x_train_multi past history : {}'.format(x_train_multi.shape))
print('\n y_train_multi Target predict : {}'.format(y_train_multi.shape))
print('x_val_multi past history : {}'.format(x_val_multi.shape))
print('\n y_val_multi Target predict : {}'.format(y_val_multi.shape))


train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))


def kcp_Finance_RNNModel():

    kcp_fn_model = tf.keras.models.Sequential()
    kcp_fn_model.add(tf.keras.layers.GRU(64, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
    kcp_fn_model.add(tf.keras.layers.GRU(32, activation='tanh', return_sequences=True))
    kcp_fn_model.add(tf.keras.layers.SimpleRNN(128))
    kcp_fn_model.add(tf.keras.layers.Dropout(0.4))
    kcp_fn_model.add(tf.keras.layers.Dense(64, kernel_initializer='orthogonal'))
    kcp_fn_model.add(tf.keras.layers.Dropout(0.2))
    kcp_fn_model.add(tf.keras.layers.Dense(32, activation='relu'))
    kcp_fn_model.add(tf.keras.layers.Dense(1))
    kcp_fn_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    kcp_fn_model.summary()
    tf.keras.utils.plot_model(kcp_fn_model, 'kcp_finance_RNNModel.png')
    tf.keras.utils.plot_model(kcp_fn_model, 'kcp_finance_RNNModel_with_shape_info.png', show_shapes=True)

    return kcp_fn_model


kcp_model = kcp_Finance_RNNModel()

for x, y in val_data_multi.take(1):
    print(kcp_model.predict(x).shape)

log_dir = file_dir+"/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

multi_step_history = kcp_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50,
                                          callbacks=[tensorboard_callback])


kcp_model.save('kcp_finance_model.h5', save_format='h5')

plot_train_history(multi_step_history, 'KCP Finance Training and validation loss')

for x, y in val_data_multi.take(5):
    multi_step_plot(x[0], y[0], kcp_model.predict(x)[0])

