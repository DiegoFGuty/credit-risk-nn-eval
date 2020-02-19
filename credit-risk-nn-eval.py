import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import tensorflow.keras as kr

# Se lee la información
datos = pd.read_csv('dataset.csv', sep=";", usecols=[0, 1, 2, 3])
X = np.array(datos.values)
# print(X)

resultados = pd.read_csv('dataset.csv', sep=";", usecols=[4])
Y = np.array(resultados.values)
# print(Y)

# Se crea la red neuronal
lr = 0.01
nn = [4, 16, 8, 1]

modelo = kr.Sequential()

# Capa 1
l1 = modelo.add(kr.layers.Dense(nn[1], activation='relu'))

# Capa 2
l2 = modelo.add(kr.layers.Dense(nn[2], activation='relu'))

# Capa 3
l3 = modelo.add(kr.layers.Dense(nn[3], activation='sigmoid'))

# Se compila el modelo, definiendo la función de coste y el optimizador.
modelo.compile(loss='mse',
               optimizer=kr.optimizers.SGD(lr=0.05),
               metrics=['acc'])


# Y entrenamos al modelo.
resultado = modelo.fit(X, Y, epochs=500, batch_size=20, verbose=0)

# Hacemos una prueba
x_pred = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 1],
                   [1, 1, 0, 0],
                   [1, 1, 1, 1],
                   [1, 1, 0, 1]])
y_pred = modelo.predict(x_pred)
print(y_pred)
