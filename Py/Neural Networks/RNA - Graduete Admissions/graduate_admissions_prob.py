''' Essa rede neural foi feita para prever a probabilidade de um aluno 
    ser aprovado em uma universidade de acordo com a base de dados 
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn import metrics

base = pd.read_csv('Admission_Predict_Ver1.1.csv')
base = base.drop('Serial No.', axis=1)

previsores = base.iloc[:, 0:7].values
classe = base.iloc[:, 7].values
classe = classe.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Transformar colunas 2 e 6 em "dummy" para poder fazer o treino 
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [2,6])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores)

# Criação da rede neural 
regressor = Sequential([
        tf.keras.layers.Dense(units=7, activation = 'relu', input_dim=12),
        tf.keras.layers.Dense(units=7, activation = 'relu'),
        tf.keras.layers.Dense(units=1, activation = 'linear')])

regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, classe, batch_size = 10, epochs = 1000)

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
with open('regressor_graduate-admissions.json', 'w') as json_file:
    json_file.write(regressor_json)
regressor.save_weights('regressor_graduate-admissions.h5')
'''

# Carregar

'''
a = open('regressor_graduate-admissions.json', 'r')
network_structure = a.read()
a.close()
regressor = model_from_json(network_structure)
regressor.load_weights('regressor_graduate-admissions.h5')
'''
