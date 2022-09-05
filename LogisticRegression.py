# Libreria para lectura de datos
import pandas as pd

# Frameworks para creacion de un modelo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Libreria para graficación
import matplotlib.pyplot as plt

# Obtención los datos 
df = pd.read_excel('Raisin_Dataset.xlsx')

# Cambio de strings a int para el modelo
df["Class"].replace({"Kecimen": 0, "Besni": 1}, inplace=True)

# Se crea el modelo de regresión logistica 
model = LogisticRegression()
model

# Selección de "x" y "y" del modelo
X_raisin = df.drop(['Class'], axis = 1)  
y_raisin = df['Class']  

# Separación de en train y test
X_train, X_test, y_train, y_test = train_test_split(X_raisin, y_raisin, random_state=1)

# Ejecución del modelo
model.fit(X_train, y_train)

print("Coeficiebtes del modelo:", model.coef_)
print("Intercepción", model.intercept_)

# Prediccion  usando el modelo sobre de los x_test 
y_fit = model.predict(X_test)

# Mean Square Error de los datos predichos y los reales
print("Mean square error del modelo:", mean_squared_error(y_test, y_fit))

# Reestructuración de los datos para poder graficar
y_df_fit = pd.DataFrame(y_fit, columns = ['Predicted value'])
y_df_test = y_test.reset_index().drop('index', axis = 1)
y_df_fit['Real value'] = y_df_test
y_df_fit.reset_index(inplace=True)

# Grafica sobre datos predichos y datos reales
y_df_fit.plot(x="index", y=["Real value", "Predicted value"])
plt.show()