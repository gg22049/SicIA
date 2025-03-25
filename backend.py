import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import Dense, Dropout, BatchNormalization, LSTM, Input
from keras.src.optimizers import Adam
from keras.src.saving.saving_lib import load_model
from keras.src.models import Model

# Limpieza de Datos
# df = pd.read_csv('data_raw.csv')
# df.dropna(axis=0, how='all', inplace=True)
# df["prcp"] = df["prcp"].fillna(0)
# df.to_csv('data_cleaned.csv', index=False)

# Nomalización de datos
# df = pd.read_csv('data_cleaned.csv', header='infer', encoding='utf-8')
# df['date'] = pd.to_datetime(df['date'])
# df['month'] = df['date'].dt.month
# colm = ['month', 'tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres', 'prcp']
# df = pd.concat([df.drop(columns=['prcp']),df['prcp']],axis=1) # Cambio de columna
# df['prcp'] = np.log1p(df['prcp']) # Transformar datos
# df = df[colm]
# scaler = MinMaxScaler()
# df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=colm)
# df_normalized.to_csv('data_normalized.csv', index=False)


# Separación en semanas
# df = pd.read_csv('data_normalized.csv', header='infer', encoding='latin1')
# def secuenciar (array):
#     x,y,y_bin = [],[],[]
#     sec = 7
#     for i in range (len(array)-sec):
#         x.append(array[i:i + sec, :-1])
#         y.append(array[i + sec, -1])
#         if array[i + sec, -1] > 0:
#             y_bin.append(1)
#         else:
#             y_bin.append(0)
#     return np.array(x), np.array(y),np.array(y_bin)
#
# X,Y,Y_bin = secuenciar(df.values)

# Modelo
# epochs = 300
# bs = 8
# lr = 0.005
# neurons = 200
# outputs = 1
# dp = 0.2
# inputs_layer = Input(shape=(7,7))
#
# x = LSTM(units=neurons, return_sequences=True)(inputs_layer)
# x = BatchNormalization()(x)
# x = Dropout(dp)(x)
# x = LSTM(units=neurons, return_sequences=False)(x)
# x = BatchNormalization()(x)
# x = Dropout(dp)(x)
# prcp_prob = Dense(units=1, activation='sigmoid', name='prcp_prob')(x)
# prcp_mm = Dense(units=1, activation='linear', name='prcp_mm')(x)
# model = Model(inputs=inputs_layer, outputs=[prcp_prob, prcp_mm])
# model.compile(loss={'prcp_prob':'binary_crossentropy', 'prcp_mm':'mae'}, optimizer=Adam(learning_rate=lr), metrics={'prcp_prob':'accuracy', 'prcp_mm':'mae'})
# summary = model.fit(X, {'prcp_prob':Y_bin, 'prcp_mm':Y}, epochs=epochs, batch_size=bs, validation_split=0.2, verbose=1)
# model.save('model_alpha.keras')

# Visualizar entrenamiento
# plt.figure(figsize=(10, 4))
# plt.plot(summary.history['loss'], label='Training Loss')
# if 'val_loss' in summary.history:
#     plt.plot(summary.history['val_loss'], label='Validation Loss', linestyle='--', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Evolución de la pérdida')
# plt.savefig('loss_plot.png')
# plt.show()
#
# plt.figure(figsize=(10, 4))
# plt.plot(summary.history['prcp_mm_mae'], label='Training MAE (prcp_mm)')
# if 'val_prcp_mm_mae' in summary.history:
#     plt.plot(summary.history['val_prcp_mm_mae'], label='Validation MAE (prcp_mm)', linestyle='--', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('MAE')
# plt.legend()
# plt.title('Error Absoluto Medio en la predicción de mm de precipitación')
# plt.savefig('mae_prcp_mm_plot.png')
# plt.show()
#
# plt.figure(figsize=(10, 4))
# plt.plot(summary.history['prcp_prob_accuracy'], label='Training Accuracy (prcp_prob)')
# if 'val_prcp_prob_accuracy' in summary.history:
#     plt.plot(summary.history['val_prcp_prob_accuracy'], label='Validation Accuracy (prcp_prob)', linestyle='--', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Precisión en la predicción binaria de lluvia')
# plt.savefig('accuracy_prcp_prob_plot.png')
# plt.show()

# Prueba
# m = keras.models.load_model("model_alpha.keras")

# for i in range(1,6):
#     idx = random.randint(0, len(X) - 1)
#     entrada = X[idx].reshape(1, 7, 7)
#     prediccion = m.predict(entrada)
#
#     prcp_scaler = MinMaxScaler()
#     prcp_scaler.fit(df[['prcp']])
#     prediccion_2d = prediccion[1].reshape(-1, 1)
#     prcp_predict = prcp_scaler.inverse_transform(prediccion_2d)
#     prcp_predict = np.exp(prcp_predict)
#     y = Y[idx].reshape(-1, 1)
#     y_real = prcp_scaler.inverse_transform(y)
#     y_real = np.exp(y_real)
#
#     error = abs((prcp_predict[0][0] - y_real) / y_real) * 100 if y_real != 0 else 100
#
#     print("Prueba N",i)
#     print(f"Valor real: {y_real[0][0]} mm")
#     print(f"Precipitación predicha: {prcp_predict[0][0]} mm")
#     print(f"Error relativo: {error} %")
#     print(len(X))

# Función para main
# def test():
#     m = keras.models.load_model("model_alpha.keras")
#
#     idx = random.randint(0, len(X) - 1)
#     entrada = X[idx].reshape(1, 7, 7)
#     prediccion = m.predict(entrada)
#
#     prcp_scaler = MinMaxScaler()
#     prcp_scaler.fit(df[['prcp']])
#     prediccion_2d = prediccion[1].reshape(-1, 1)
#     prcp_predict = prcp_scaler.inverse_transform(prediccion_2d)
#     prcp_predict = np.exp(prcp_predict)
#     y = Y[idx].reshape(-1, 1)
#     y_real = prcp_scaler.inverse_transform(y)
#     y_real = np.exp(y_real)
#
#     error = abs((prcp_predict[0][0] - y_real[0][0]) / y_real[0][0]) * 100 if y_real != 0 else 100
#
#     return y_real[0][0], prcp_predict[0][0], error