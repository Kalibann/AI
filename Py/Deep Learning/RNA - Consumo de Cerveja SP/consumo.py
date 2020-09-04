"""
    Essa rede neural foi feita para prever o consumo de cerveja em SP de
    acordo com a base de dados utilizada.
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json 
from sklearn import metrics

# Importação e tratamento dos dados
base = pd.read_csv('Consumo_cerveja.csv')
base = base.dropna()
base = base.drop('Data', axis=1)

base.columns = base.columns.str.replace(' ', '_')
base.columns = base.columns.str.replace('[)(]' ,'')

base['Temperatura_Media_C'] = base['Temperatura_Media_C'].astype(str).str.replace(',','.').astype(float)
base['Temperatura_Minima_C'] = base['Temperatura_Minima_C'].astype(str).str.replace(',','.').astype(float)
base['Temperatura_Maxima_C'] = base['Temperatura_Maxima_C'].astype(str).str.replace(',','.').astype(float)
base['Precipitacao_mm'] = base['Precipitacao_mm'].astype(str).str.replace(',','.').astype(float)
base['Final_de_Semana'] = base['Final_de_Semana'].astype(int)

dummy = pd.get_dummies(base.Final_de_Semana)
base = base.drop('Final_de_Semana', axis=1)
base = pd.concat([base, dummy], axis=1)

previsores = base.iloc[:, [0,1,2,3, 5,6]].values
classe = base.iloc[:, 4].values
classe = classe.reshape(-1,1)

# Criação da rede neural
regressor = Sequential([
        tf.keras.layers.Dense(units=4, activation = 'relu', input_dim=6),
        tf.keras.layers.Dense(units=4, activation = 'relu'),
        tf.keras.layers.Dense(units=1, activation = 'linear')])

regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, classe, batch_size = 5, epochs = 1000)

predict = regressor.predict(previsores)

# Métricas para analisar o desempenho da rede neural
media_real = classe.mean()
media_previsoes = predict.mean()

print('\nMAE:', metrics.mean_absolute_error(classe, predict))
print('MSE:', metrics.mean_squared_error(classe, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(classe, predict)))

# Salvar
'''
regressor_json = regressor.to_json()
with open('regressor_consumo.json', 'w') as json_file:
    json_file.write(regressor_json)
regressor.save_weights('regressor_consumo.h5')
'''

# Carregar

'''
a = open('regressor_consumo.json', 'r')
network_structure = a.read()
a.close()
regressor = model_from_json(network_structure)
regressor.load_weights('regressor_consumo.h5')
'''