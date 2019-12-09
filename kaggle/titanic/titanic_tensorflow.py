import tensorflow as tf
import numpy as np
import pandas as pd

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
# titanic_test_df = pd.read_csv("./titanic_data/test.csv")
# y_test_df = pd.read_csv("./titanic_data/gender_submission.csv")

tf.keras.backend.set_floatx('float32')

tr_data = transform_data(titanic_df)
y_data = tr_data.pop('Survived')


dataset = tf.data.Dataset.from_tensor_slices((tr_data.values, y_data.values))
train_ds = dataset.shuffle(len(titanic_df)).batch(50)
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