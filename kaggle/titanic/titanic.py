'''
https://www.kaggle.com/c/titanic/data

PassengerId     int64   : 탑승자 일련번호
Survived        int64   : 생존 여부( 0 = 사망 / 1 = 생존)
Pclass          int64   : 선실 등급 ( 1 = 일등석 / 2 = 이등석 / 3 = 삼등석)
Name            object  : 성별
Sex             object  : 이름
Age             float64 : 나이
SibSp           int64   : 같이 탑승한 형제/자매/배우자 인원수
Parch           int64   : 같이 탑승한 부모님/어린이 인원수
Ticket          object  : 티켓번호
Fare            float64 : 요금
Cabin           object  : 선실번호
Embarked        object  : 중간 정착 항구 ( C = Cherbourg, Q = Queenstown, S = Southampton)
dtypes: float64(2), int64(5), object(5)

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn import preprocessing
import tensorflow as tf

print(tf.__version__)
tf.keras.backend.clear_session()

def change_null(data):
    # 결측지 제거
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Cabin'].fillna('N', inplace=True)
    data['Embarked'].fillna('N', inplace=True)
    return data


def encode_label(data):
    data['Cabin'] = data['Cabin'].str[:1]
    label_list = ['Cabin', 'Sex', 'Embarked']

    for label in label_list:
        data[label] = pd.Categorical(data[label])
        data[label] = data[label].cat.codes
    return data


def drop_data(data):
    data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return data


def transform_data(data):
    data = change_null(data)
    data = encode_label(data)
    data = drop_data(data)
    return data


def get_change_age(age):
    if age <= 7:
        return 7
    elif age <= 12:
        return 12
    elif age <= 19:
        return 17
    elif age <= 29:
        return 20
    elif age <= 39:
        return 30
    elif age <= 49:
        return 40
    elif age <= 59:
        return 50
    elif age <= 69:
        return 60
    else:
        return 70


titanic_df = pd.read_csv("./titanic_data/train.csv")
titanic_test_df = pd.read_csv("./titanic_data/test.csv")
y_test_df = pd.read_csv("./titanic_data/gender_submission.csv")

print(titanic_df.head())
print(titanic_df.info())
print(titanic_df.isnull().sum())
print(titanic_df.info())


print("Sex : ", titanic_df['Sex'].value_counts())
print("Age : ", titanic_df['Age'].value_counts())
print("Cabin : ", titanic_df['Cabin'].value_counts())
print("Embarked : ", titanic_df['Embarked'].value_counts())

print(titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())

sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.show()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)
plt.show()

titanic_df['Age_ch'] = titanic_df['Age'].apply(lambda x : get_change_age(x))
print(titanic_df.groupby(['Age_ch', 'Survived'])['Survived'].count())
sns.countplot(x='Age_ch',  hue='Survived',  data=titanic_df)
plt.show()
sns.barplot(x='Age_ch', y='Survived', hue='Pclass', data=titanic_df)
plt.show()
sns.barplot(x='Age_ch', y='Survived', hue='Sex', data=titanic_df)
plt.show()
titanic_df.drop('Age_ch', axis=1, inplace=True)

tf.keras.backend.set_floatx('float64')

tr_data = transform_data(titanic_df)
y_data = tr_data.pop('Survived')
test_data = transform_data(titanic_test_df)
test_y_data = y_test_df.pop('Survived')

dataset = tf.data.Dataset.from_tensor_slices((tr_data.values, y_data.values))
validate_data = dataset.shuffle(10000).batch(50).prefetch(1)


def get_compiled_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, kernel_initializer='uniform', activation='relu', input_shape=(8,)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # 커널을 랜덤한 직교 행렬로 초기화한 선형 활성화 층:
    # model.add(tf.keras.layers.Dense(64, kernel_initializer='orthogonal'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='out_layer'))

    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    
    model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=256))
    model.add(tf.keras.layers.LSTM(128))
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_crossentropy])
                  
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    '''

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.001, beta_2=0.999, amsgrad=False),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


model = get_compiled_model()

model.summary()
tf.keras.utils.plot_model(model, 'my_titanic_model.png')
tf.keras.utils.plot_model(model, 'my_titanic_model_with_shape_info.png', show_shapes=True)


class LearningRateScheduler(tf.keras.callbacks.Callback):

  def __init__(self, schedule):
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')

    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    scheduled_lr = self.schedule(epoch, lr)
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))



LR_SCHEDULE = [
    (3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)
]

def lr_schedule(epoch, lr):
  if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
    return lr
  for i in range(len(LR_SCHEDULE)):
    if epoch == LR_SCHEDULE[i][0]:
      return LR_SCHEDULE[i][1]
  return lr


callbacks = [
  tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_loss'),
  # tf.keras.callbacks.TensorBoard(log_dir='./logs')
  # LearningRateScheduler(lr_schedule)
]
model.fit(tr_data.values, y_data.values, batch_size=128, epochs=100, callbacks=callbacks,
          validation_data=validate_data)

model.evaluate(test_data.values, test_y_data.values, batch_size=10)

test_ds = tf.data.Dataset.from_tensor_slices((test_data.values, test_y_data.values)).batch(50).prefetch(1)
model.evaluate(test_ds, steps=3)

result = model.predict(test_ds)
# print(result)

json_string = model.to_json()
print(json_string)


# 가중치를 텐서플로의 체크포인트 파일로 저장합니다.
# model.save_weights('./weights/my_model')
# 모델의 상태를 복원합니다.
# 모델의 구조가 동일해야 합니다.
# model.load_weights('./weights/my_model')
# 가중치를 HDF5 파일로 저장합니다.
# model.save_weights('my_titanic_model.h5', save_format='h5')
# 모델의 상태를 복원합니다.
# model.load_weights('my_titanic_model.h5')
# 가중치와 옵티마이저를 포함하여 정확히 같은 모델을 다시 만듭니다.
# model = tf.keras.models.load_model('my_titanic_model.h5')
