import pandas as pd
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

#returns a normalised dataframe and the scaler that has been used to transform the data
def format_pv_data(pv_data_raw: pd.DataFrame):
    #format the time column
    pv_data_raw.rename(columns={pv_data_raw.columns[0]: "PeriodEnd"}, inplace=True)
    pv_data_raw['PeriodEnd'] = pd.to_datetime(pv_data_raw.PeriodEnd)
    pv_data_raw['PeriodEnd'] = pv_data_raw['PeriodEnd'].dt.strftime('%Y-%m-%d %H:%M:%S')

    #We replace all NaN values with the previous observation
    pv_data_raw = pv_data_raw.ffill()

    period_end_col = pv_data_raw.iloc[:,0]
    pv_data_vals = pv_data_raw.iloc[:,1].to_frame()

    #normalise the data
    scaler_pv = MinMaxScaler(feature_range=(0, 1))
    scaled_pv_data = pd.DataFrame(scaler_pv.fit_transform(pv_data_vals))

    return scaler_pv, scaled_pv_data.join(period_end_col)

def format_weather_data(weather_data_raw: pd.DataFrame):
    #format the time column
    weather_data_raw = weather_data_raw.drop(["PeriodStart"], axis=1)
    weather_data_raw['PeriodEnd'] = pd.to_datetime(weather_data_raw.PeriodEnd)
    weather_data_raw['PeriodEnd'] = weather_data_raw['PeriodEnd'].dt.strftime('%Y-%m-%d %H:%M:%S')

    period_end_col = weather_data_raw.iloc[:, 0]
    weather_data_vals = weather_data_raw.iloc[:, 1:]

    #normalise the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_weather_data = pd.DataFrame(scaler.fit_transform(weather_data_vals))

    return scaled_weather_data.join(period_end_col)

def create_dataset(pv_data: pd.DataFrame, weather_data: pd.DataFrame):
    #we predict the next 7 days of pv data using weather data from the past 4 weeks and pv data from the past 7 days
    X_weather_list, X_pv_list, y_list = list(), list(), list()

    for i in range(len(weather_data)):
        #each training sample includes 1344 weather data readings
        weather_end_index = i + 1344
        if weather_end_index > len(weather_data):
            break
        weather_sample = weather_data[i:weather_end_index].drop(["PeriodEnd"], axis=1).to_numpy()

        #find the index of the row in the pv_data corresponding to the final entry in the weather_data
        #we assume this to be the current day and want to predict the pv for the next 7 days
        date_end_weather = weather_data[weather_end_index - 1:weather_end_index].iloc[0]["PeriodEnd"]
        pv_index_final_weather_date = pv_data[pv_data.PeriodEnd == date_end_weather].index.tolist()

        #check that the date is in the pv_dataframe (start of weather recordings is 6 months earlier than the start
        #of the pv recordings)
        if len(pv_index_final_weather_date) == 0:
            continue

        pv_start_index = pv_index_final_weather_date[0] + 1
        pv_end_index = pv_start_index + 10080
        if pv_end_index > len(pv_data):
            break

        #now construct the input pv sample from the previous 7 days of pv data
        input_pv_start_index = pv_start_index - 10080
        if input_pv_start_index < 0:
            continue

        pv_sample = pv_data[pv_start_index:pv_end_index].drop(["PeriodEnd"], axis=1).to_numpy()
        input_pv_sample = pv_data[input_pv_start_index:pv_start_index].drop(["PeriodEnd"], axis=1).to_numpy()

        X_weather_list.append(weather_sample)
        X_pv_list.append(input_pv_sample)
        y_list.append(pv_sample)

    # the X data has the shape [samples, timesteps, features] while the y data has shape [samples, timesteps]
    X_weather = np.array(X_weather_list)
    X_pv = np.array(X_pv_list)
    y = np.squeeze(np.array(y_list))

    return X_weather, X_pv, y

#create a model with two input lstm layers and a single dense output layer
def create_model():
    input_shape_pv = (10080, 1)
    input_shape_weather = (1344, 15)

    input_pv = layers.Input(shape=input_shape_pv)
    input_weather = layers.Input(shape=input_shape_weather)

    lstm_pv = layers.LSTM(units=100, activation='elu')(input_pv)
    lstm_pv = layers.BatchNormalization()(lstm_pv)

    lstm_weather = layers.LSTM(units=100, activation='elu')(input_weather)
    lstm_weather = layers.BatchNormalization()(lstm_weather)

    concatenated = layers.Concatenate()([lstm_pv, lstm_weather])
    output = layers.Dense(units=10080, activation='relu')(concatenated)

    return keras.Model(inputs=[input_weather, input_pv], outputs=output)

def train_test_split(X_weather: np.array, X_pv: np.array, y: np.array, train_samples: int, test_samples: int):
    test_index = -1 * test_samples
    X_weather_train = np.array(X_weather)[:train_samples, :, :]
    X_weather_test = np.array(X_weather)[test_index:, :, :]
    X_pv_train = np.array(X_pv)[:train_samples, :, :]
    X_pv_test = np.array(X_pv)[test_index:, :, :]
    X_train = [X_weather_train, X_pv_train]
    X_test = [X_weather_test, X_pv_test]

    y_train = np.array(y)[:train_samples, :]
    y_test = np.array(y)[test_index:, :]
    return X_train, X_test, y_train, y_test

def fit_model(model, X_train, y_train):
    opt = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.001)
    model.compile(loss="mae", optimizer=opt)
    model.fit(X_train, y_train, batch_size=128, epochs=20, shuffle=True)
    return model


def predict(model, X_test, y_test, scaler_pv):
    yhat = model.predict(X_test)
    inv_yhat = scaler_pv.inverse_transform(yhat)
    inv_ytest = scaler_pv.inverse_transform(y_test)

    # calculate mean absolute error
    return np.mean(keras.losses.mean_absolute_error(inv_ytest, inv_yhat))

#pv data is colleted every minute from 1st March 2023 to 31st August 2023
pv_data_raw = pd.read_csv(r"C:\Users\jasmi\PycharmProjects\SwitchDin-data\pv_data.csv")
scaler_pv, pv_data = format_pv_data(pv_data_raw)

#weather data is data collected every half an hour from 14th Sep 2022 to 17 Sep 2023
weather_data_raw = pd.read_csv(r"C:\Users\jasmi\PycharmProjects\SwitchDin-data\weather_data.csv")
weather_data = format_weather_data(weather_data_raw)

X_weather, X_pv, y = create_dataset(pv_data, weather_data)
X_train, X_test, y_train, y_test = train_test_split(X_weather, X_pv, y, 3000, 160)

model = create_model()
model = fit_model(model, X_train, y_train)
"""
Epoch 1/20
24/24 [==============================] - 799s 33s/step - loss: 0.1398
Epoch 2/20
24/24 [==============================] - 1050s 44s/step - loss: 0.1196
Epoch 3/20
24/24 [==============================] - 1151s 48s/step - loss: 0.0982
Epoch 4/20
24/24 [==============================] - 816s 34s/step - loss: 0.0795 
Epoch 5/20
24/24 [==============================] - 853s 35s/step - loss: 0.0675
Epoch 6/20
24/24 [==============================] - 900s 37s/step - loss: 0.0620
Epoch 7/20
24/24 [==============================] - 933s 39s/step - loss: 0.0591 
Epoch 8/20
24/24 [==============================] - 964s 40s/step - loss: 0.0579 
Epoch 9/20
24/24 [==============================] - 924s 38s/step - loss: 0.0563
Epoch 10/20
24/24 [==============================] - 1089s 45s/step - loss: 0.0555 
Epoch 11/20
24/24 [==============================] - 954s 39s/step - loss: 0.0546 
Epoch 12/20
24/24 [==============================] - 885s 37s/step - loss: 0.0545 
Epoch 13/20
24/24 [==============================] - 954s 40s/step - loss: 0.0535 
Epoch 14/20
24/24 [==============================] - 831s 35s/step - loss: 0.0537 
Epoch 15/20
24/24 [==============================] - 807s 33s/step - loss: 0.0525
Epoch 16/20
24/24 [==============================] - 798s 33s/step - loss: 0.0521
Epoch 17/20
24/24 [==============================] - 751s 31s/step - loss: 0.0524
Epoch 18/20
24/24 [==============================] - 802s 33s/step - loss: 0.0520
Epoch 19/20
24/24 [==============================] - 756s 31s/step - loss: 0.0516
Epoch 20/20
24/24 [==============================] - 732s 30s/step - loss: 0.0516
"""

mae = predict(model, X_test, y_test, scaler_pv)
#mae = 2.603632