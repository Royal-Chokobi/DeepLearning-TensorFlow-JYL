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

            print("날짜 : {}, 종가 : {}, 전일비 : {}, 시가 : {}, 고가 : {}, 저가 : {}, 거래량 : {}".format(fin_date, adj_Close, adj_won, open_pri, high_pri, low_pri, adj_mount))
            finance_list.append([fin_date, adj_Close, adj_won, open_pri, high_pri, low_pri, adj_mount])


# finance_array = np.array(finance_list)
# finance_df = pd.DataFrame(np.array(finance_array),columns=['Date', 'Adj_Close', 'Adj_Won', 'Open', 'High', 'Low', 'Volume'])
# print(finance_df.head())
#
# finance_df.to_csv("finance.csv", mode='w', header=True, index=False)

finance_data = pd.read_csv('finance.csv')
high_prices = finance_data['High'].values
low_prices = finance_data['Low'].values
mid_prices = (high_prices + low_prices) / 2

print(finance_data.dtypes)
print(finance_data.head())

plt.plot(finance_data['Adj_Close'].values)
plt.plot(finance_data['High'].values)
plt.plot(finance_data['Low'].values)
plt.show()

y_data = finance_data.pop('Adj_Close')
tr_data = finance_data


dataset = tf.data.Dataset.from_tensor_slices((tr_data.values, y_data.values))
train_ds = dataset.shuffle(len(finance_data)).batch(50)
def get_compiled_model():

    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(10, activation='relu'))
    # model.add(tf.keras.layers.Dense(10, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = get_compiled_model()
model.fit(train_ds, epochs=100)
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
