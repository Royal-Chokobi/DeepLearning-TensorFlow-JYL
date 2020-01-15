import pandas as pd
import numpy as np
import os
import tensorflow as tf

kcp_fn_model = tf.keras.models.load_model('kcp_finance_model.h5')
kcp_fn_model.load_weights('kcp_finance_model.h5')

print(kcp_fn_model.summary())


file_dir = os.path.dirname(os.path.realpath(__file__))
finance_data = pd.read_csv(file_dir+'/finance_predict_data.csv')
print(finance_data.head())

fin_real_date = finance_data.pop('realDate')

data_len = len(finance_data)
fn_date = finance_data.pop('Date')

finance_df = finance_data
finance_df.index = fin_real_date

dataset = finance_df.values
data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)
price_mean = dataset[15:, 0].mean(axis=0)
price_std = dataset[15:, 0].std(axis=0)
dataset = (dataset-data_mean)/data_std

data = [dataset]
x_data = np.array(data)
predict_data = kcp_fn_model.predict(x_data)[0]
predict_Adj_Close = (predict_data[0]*price_std)+price_mean

print("내일 예측 주식 종가 : {}".format(predict_Adj_Close))
