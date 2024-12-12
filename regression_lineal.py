#Bibliotecas a utilizar 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar el conjunto de datos desde 'Jalisco.csv'
mydataset = pd.read_csv('Jalisco.csv')

X_multiple = mydataset.iloc[:, [1, 3, 4]]
y_multiple = mydataset['Cosechada']

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size=0.20, random_state=42)

# Definir el modelo de regresión lineal
lr_multiple = LinearRegression()

# Entrenar el modelo
lr_multiple.fit(X_train, y_train)

# Realizar una predicción
Y_pred_multiple = lr_multiple.predict(X_test)

# Imprimir los coeficientes y la precisión del modelo
print("DATOS DEL MODELO REGRESIÓN LINEAL MÚLTIPLE")
print("Valor de las pendientes o coeficientes -a- :")
print(lr_multiple.coef_)
print("Valor de la intersección o coeficiente -b-:")
print(lr_multiple.intercept_)
print("Precisión del modelo en el conjunto de entrenamiento:")
print(lr_multiple.score(X_train, y_train))

# Graficar los datos reales y la línea de regresión
plt.scatter(X_test.iloc[:, 0], y_test, color='blue', label='Datos reales')
plt.plot(X_test.iloc[:, 0], Y_pred_multiple, color='red', linewidth=2, label='Regresión lineal')
plt.xlabel('Variable X')
plt.ylabel('Cosechada')
plt.legend()
plt.show()

# Crear un DataFrame con las predicciones
df_predicciones = pd.DataFrame({'Real': y_test, 'Predicciones': Y_pred_multiple})

# Imprimir el DataFrame de predicciones
print(df_predicciones)

