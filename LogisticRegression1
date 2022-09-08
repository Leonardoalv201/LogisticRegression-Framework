# Libreria para lectura de datos
import pandas as pd

# Frameworks para creacion de un modelo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Libreria para graficación
import seaborn as sns
import matplotlib.pyplot as plt #graphics

# Obtención los datos 
df = pd.read_excel('Raisin_Dataset.xlsx')

# Cambio de strings a int para el modelo
df["Class"].replace({
    "Kecimen": 0,
    "Besni": 1
    }, inplace=True)

# Se crea el modelo de regresión logistica 
model = LogisticRegression(penalty = 'l1', solver = 'saga')

# Selección de "x" y "y" del modelo
X_raisin = df.drop(['Class'], axis = 1)  
y_raisin = df['Class']  

# Separación de en train y test
X_train, X_test, y_train, y_test = train_test_split(X_raisin, y_raisin, random_state=1)

# Ejecución del modelo
model.fit(X_train, y_train)

print("Coeficiebtes del modelo:", model.coef_, '\n')
print("Intercepción", model.intercept_, '\n', '\n')

#Prediccion del modelo en datos de train
y_pred_train = model.predict(X_train)

# Mean Square Error de los datos predichos y los reales
print("Mean square error del modelo en datos train:", mean_squared_error(y_train, y_pred_train), '\n')

# Accuracy del modelo
print("Accuracy del modelo en datos train:", accuracy_score(y_train, y_pred_train), '\n', '\n')

# Prediccion  usando el modelo sobre de los x_test 
y_pred = model.predict(X_test)

# Mean Square Error de los datos predichos y los reales
print("Mean square error del modelo en datos test:", mean_squared_error(y_test, y_pred), '\n')

# Accuracy del modelo
print("Accuracy del modelo en datos test:", accuracy_score(y_test, y_pred), '\n', '\n')

#Matriz de confusion de los datos
cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred), columns = ['Yes','No'])
cm_df[''] = ['Yes', 'No']
cols = cm_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
cm_df = cm_df[cols]
cm_df = cm_df.set_index('')
print('Confusion matrix: \n')
print(cm_df, '\n', '\n')

#Heat map de los datos obtenidos contra los reales
cm_df = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.show()

#Dummy data para validacion del modelo
import random
dummy_data = [
    [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)],
     [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)],
     [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)],
     [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)],
     [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)],
     [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)],
     [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)],
     [random.randint(25387, 235047),
     random.uniform(225, 997),
     random.uniform(143, 492),
     random.uniform(0, 1),
     random.randint(26139, 278217),
     random.uniform(0, 1),
     random.uniform(619, 2697)]
]

#Pasamos nuestros dummy_data a DataFrame para correr un predict
df = pd.DataFrame(dummy_data)

#Predict de los datos con el modelo
y_pred_dummy = model.predict(df)

print("Prediccion sobre dummy data", y_pred_dummy , '\n', '\n')

'''
Mejora del modelo

Cambiamos solved de 'saga' a 'liblinear' 
'''

#Creamos un nueov modelo con el nuevo solver
model2 = LogisticRegression(penalty = 'l1', solver = 'liblinear')

#Entrenamosnuestro modelo con los train data
model2.fit(X_train, y_train)

#Predecimos nuestros datos de train para conocer el error y accuracy
y_pred2_train = model2.predict(X_train)


print("Accuracy del modelo 2 en train data:", accuracy_score(y_train, y_pred2_train), '\n')
print("Mean square error del modelo 2 en train data:", mean_squared_error(y_train, y_pred2_train), '\n', '\n')

#Predecimos nuestros datos de test 
y_pred2 = model2.predict(X_test)

print("Accuracy del modelo 2:", accuracy_score(y_test, y_pred2), '\n')
print("Mean square error del modelo 2:", mean_squared_error(y_test, y_pred2), '\n', '\n')

#Variables para conocer la mejora del modelo
upgrade = accuracy_score(y_test, y_pred2) * 100 / accuracy_score(y_test, y_pred)
upgrade_error = mean_squared_error(y_test, y_pred2) * 100 / mean_squared_error(y_test, y_pred)

print('Nuestro modelo es', upgrade_error, '% mejor', '\n')
print('Nuestro modelo es', upgrade, '% mejor', '\n', '\n')

cm_df = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.show()