import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import urllib
import time
import datetime
from dateutil.parser import parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
import os
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK']='True'
print(tf.__version__)
tf.keras.backend.clear_session()

stockItem = '060250'

url = 'http://finance.naver.com/item/sise_day.nhn?code='+ stockItem
html = urlopen(url)
source = BeautifulSoup(html.read(), "html.parser")

maxPage = source.find_all("table",align="center")
mp = maxPage[0].find_all("td",class_="pgRR")
mpNum = int(mp[0].a.get('href')[-3:])

finance_list = []

for page in range(mpNum, 0, -1):
    break
    print (str(page) )
    url = 'http://finance.naver.com/item/sise_day.nhn?code=' + stockItem +'&page='+ str(page)
    html = urlopen(url)
    source = BeautifulSoup(html.read(), "html.parser")
    srlists=source.find_all("tr")
    isCheckNone = None

    # if((page % 1) == 0):
    #     time.sleep(1.50)

    for i in range(len(srlists)-1, 0, -1):
        if(srlists[i].span != isCheckNone):

            # srlists[i].td.text
            date = srlists[i].find_all("td",align="center")[0].text
            fin_date = int( srlists[i].find_all("td",align="center")[0].text.replace('.','') )
            adj_Close = float( srlists[i].find_all("td",class_="num")[0].text.replace(',','') )
            adj_won =  float( srlists[i].find_all("td",class_="num")[1].text.replace(',','') )
            if adj_won > 0:
                if str(srlists[i].find_all("td",class_="num")[1].find('img').get('alt')) == '하락':
                    adj_won = adj_won * -1
            open_pri = float( srlists[i].find_all("td",class_="num")[2].text.replace(',','') )
            high_pri = float( srlists[i].find_all("td",class_="num")[3].text.replace(',','') )
            low_pri = float( srlists[i].find_all("td",class_="num")[4].text.replace(',','') )
            adj_mount = float( srlists[i].find_all("td",class_="num")[5].text.replace(',','') )

            print("날짜 : {}, 종가 : {}, 전일비 : {}, 시가 : {}, 고가 : {}, 저가 : {}, 거래량 : {}".format(date, adj_Close, adj_won, open_pri, high_pri, low_pri, adj_mount))
            finance_list.append([date, fin_date, adj_Close, adj_won, open_pri, high_pri, low_pri, adj_mount])


# finance_array = np.array(finance_list)
# finance_df = pd.DataFrame(np.array(finance_array),columns=['realDate', 'Date', 'Adj_Close', 'Adj_Won', 'Open', 'High', 'Low', 'Volume'])
# print(finance_df.head())
#
# finance_df.to_csv("finance.csv", mode='w', header=True, index=False)

finance_data = pd.read_csv('finance.csv')
high_prices = finance_data['High'].values
low_prices = finance_data['Low'].values
mid_prices = (high_prices + low_prices) / 2

print(finance_data.dtypes)
print(finance_data.head())


# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원


def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


uni_data = finance_data['Adj_Close']
uni_data.index = finance_data['realDate']
uni_data.plot(subplots=True)
plt.show()

ax = plt.gca()
HL_data = finance_data
HL_data.plot(kind='line', x='realDate',y='Low',ax=ax)
HL_data.plot(kind='line', x='realDate',y='High', color='red', ax=ax)
HL_data.index = finance_data['realDate']
HL_data.plot(subplots=True)
plt.show()

wonUpDown_data = finance_data['Adj_Won']
wonUpDown_data.index = finance_data['realDate']
wonUpDown_data.plot(subplots=True)
plt.show()


fin_real_date = finance_data.pop('realDate')

data_len = len(finance_data)
fn_date = finance_data.pop('Date')
fn_volume = finance_data.pop('Volume')
# fn_Adj_Won = finance_data.pop('Adj_Won')

finance_price = min_max_scaling(finance_data)
# finance_date = min_max_scaling(fn_date).reshape(data_len, 1)
finance_volume = min_max_scaling(fn_volume).reshape(data_len, 1)
# finance_Adj_Won = min_max_scaling(fn_Adj_Won).reshape(data_len, 1)

x = np.concatenate([finance_price, finance_volume], axis=1)
# x.index = fin_real_date

features_columns = ['Adj_Close', 'Adj_Won', 'Open', 'High', 'Low', 'Volume']
finance_df = pd.DataFrame(x, columns=features_columns, index=fin_real_date)
print(finance_df.head())

y_data = finance_price[:, [0]]

dataX = []
dataY = []
seq_length = 5

for i in range(0, len(y_data) - seq_length):
    _x = x[i: i+seq_length]
    _y = y_data[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size

x_train = np.array(dataX[0:train_size])
y_train = np.array(dataY[0:train_size])

x_test = np.array(dataX[train_size:len(dataX)])
y_test = np.array(dataY[train_size:len(dataY)])

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = dataset.batch(50)

print(dataset)
print(train_ds)


print("="*200)

TRAIN_SPLIT = 1000
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

shape_uni_data = np.asarray(uni_data).reshape(data_len, 1)


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)

        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(shape_uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(shape_uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

#
#
def create_time_steps(length):
    return list(range(-length, 0))
#
#
# def show_plot(plot_data, delta, title):
#     labels = ['History', 'True Future', 'Model Prediction']
#     marker = ['.-', 'rx', 'go']
#     time_steps = create_time_steps(plot_data[0].shape[0])
#     if delta:
#         future = delta
#     else:
#         future = 0
#
#     plt.title(title)
#     for i, x in enumerate(plot_data):
#         if i:
#             plt.plot(future, plot_data[i], marker[i], markersize=10,
#                      label=labels[i])
#         else:
#             plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
#     plt.legend()
#     plt.xlim([time_steps[0], (future+5)*2])
#     plt.xlabel('Time-Step')
#     plt.show()
#     return plt
#
#
# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Finance Sample')
#
#
# def baseline(history):
#     return np.mean(history)
#
#
# show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
#           'Baseline Prediction Example')
#
BATCH_SIZE = 100
BUFFER_SIZE = 1000
#
# train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
# train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#
# val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
# val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
#
# simple_lstm_model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
#     tf.keras.layers.Dense(1)
# ])
#
# simple_lstm_model.compile(optimizer='adam', loss='mae')
#
# for x, y in val_univariate.take(1):
#     print(simple_lstm_model.predict(x).shape)
#
#
EVALUATION_INTERVAL = 200
EPOCHS = 1
#
# simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
#                       steps_per_epoch=EVALUATION_INTERVAL,
#                       validation_data=val_univariate, validation_steps=50)
#
#
# for x, y in val_univariate.take(3):
#     plot = show_plot([x[0].numpy(), y[0].numpy(),
#                       simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
#     plot.show()


print("="*200)

# 'realDate', 'Date', 'Adj_Close', 'Adj_Won', 'Open', 'High', 'Low', 'Volume'


finance_df.plot(subplots=True)
plt.show()

TRAIN_SPLIT = 3000
dataset = finance_df.values
tr_data = dataset
data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)

dataset = (dataset-data_mean)/data_std

# print(finance_df.head())


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step):
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


for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))


def kcp_Finance_RNNModel():

    kcp_fn_model = tf.keras.models.Sequential()
    kcp_fn_model.add(tf.keras.layers.GRU(64, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
    kcp_fn_model.add(tf.keras.layers.GRU(32, activation='relu'))
    kcp_fn_model.add(tf.keras.layers.Dense(126, activation='relu'))
    kcp_fn_model.add(tf.keras.layers.Dropout(0.2))
    kcp_fn_model.add(tf.keras.layers.Dense(64, kernel_initializer='orthogonal'))
    kcp_fn_model.add(tf.keras.layers.Dropout(0.3))
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

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

multi_step_history = kcp_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50,
                                          callbacks=[tensorboard_callback])


kcp_model.save('kcp_finance_model.h5', save_format='h5')


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


plot_train_history(multi_step_history, 'KCP Finance Training and validation loss')


for x, y in val_data_multi.take(5):
    multi_step_plot(x[0], y[0], kcp_model1.predict(x)[0])




