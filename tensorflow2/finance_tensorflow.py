import datetime as dt
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import urllib
import time
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


finance_data.pop('realDate')

data_len = len(finance_data)
fn_date = finance_data.pop('Date')
fn_volume = finance_data.pop('Volume')
fn_Adj_Won = finance_data.pop('Adj_Won')

finance_price = min_max_scaling(finance_data)
finance_date = min_max_scaling(fn_date).reshape(data_len, 1)
finance_volume = min_max_scaling(fn_volume).reshape(data_len, 1)
finance_Adj_Won = min_max_scaling(fn_Adj_Won).reshape(data_len, 1)

x = np.concatenate([finance_price, finance_date, finance_volume, finance_Adj_Won], axis=1)
print(x.shape)

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

TRAIN_SPLIT = 4000
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



def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    plt.show()
    return plt


show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Finance Sample')


def baseline(history):
    return np.mean(history)


show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
          'Baseline Prediction Example')

BATCH_SIZE = 256
BUFFER_SIZE = 1000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)


EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)


for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()

# inputs = np.random.random([32, 10, 8]).astype(np.float32)
# lstm = tf.keras.layers.LSTM(4)
#
# output = lstm(inputs)  # The output has shape `[32, 4]`.
#
# lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
#
# # whole_sequence_output has shape `[32, 10, 4]`.
# # final_memory_state and final_carry_state both have shape `[32, 4]`.
# whole_sequence_output, final_memory_state, final_carry_state = lstm(inputs)

# x = np.concatenate((np.array(finance_price), np.array(finance_date)), axis=1) # axis=1, 세로로 합친다

# norm_price = min_max_scaling(price)
#
# print("price.shape: ", price.shape)
# print("price[0]: ", price[0])
# print("norm_price[0]: ", norm_price[0])
# print("="*100) # 화면상 구분용

# y_data = finance_data.pop('Adj_Close')
# tr_data = finance_data

# dataset = tf.data.Dataset.from_tensor_slices((tr_data.values, y_data.values))
# train_ds = dataset.shuffle(len(finance_data)).batch(50)
# def get_compiled_model():
#
#     # model = tf.keras.Sequential()
#     # model.add(tf.keras.layers.Dense(10, activation='relu'))
#     # model.add(tf.keras.layers.Dense(10, activation='relu'))
#     # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
#
# model = get_compiled_model()
# model.fit(train_ds, epochs=100)
#
# def get_compiled_model():
#
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Dense(10, kernel_initializer='uniform', activation='relu', input_shape=(8,)))
#     model.add(tf.keras.layers.Dense(128, activation='relu'))
#     # 커널을 랜덤한 직교 행렬로 초기화한 선형 활성화 층:
#     # model.add(tf.keras.layers.Dense(64, kernel_initializer='orthogonal'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(256, activation='relu'))
#     # model.add(tf.keras.layers.BatchNormalization())
#
#     model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='out_layer'))
#
#     '''
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#
#     inputs = keras.Input(shape=(784,), name='digits')
#     x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
#     x = layers.Dense(64, activation='relu', name='dense_2')(x)
#     outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
#
#     model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=256))
#     model.add(tf.keras.layers.LSTM(128))
#
#     model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
#                   loss=tf.keras.losses.binary_crossentropy,
#                   metrics=[tf.keras.metrics.binary_crossentropy])
#
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     '''
#
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.001, beta_2=0.999, amsgrad=False),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
