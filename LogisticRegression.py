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

# Obtención los datos 
df = pd.read_excel('Raisin_Dataset.xlsx')

# Cambio de strings a int para el modelo
df["Class"].replace({
    "Kecimen": 0,
    "Besni": 1
    }, inplace=True)

# Se crea el modelo de regresión logistica 
model = LogisticRegression(penalty = 'l1', solver = 'saga')
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

#Prediccion del modelo en datos de train
y_pred_train = model.predict(X_train)

# Mean Square Error de los datos predichos y los reales
print("Mean square error del modelo en datos train:", mean_squared_error(y_train, y_pred_train))
# Accuracy del modelo
print("Accuracy del modelo en datos train:", accuracy_score(y_train, y_pred_train))

# Prediccion  usando el modelo sobre de los x_test 
y_pred = model.predict(X_test)

# Mean Square Error de los datos predichos y los reales
print("Mean square error del modelo en datos test:", mean_squared_error(y_test, y_pred))

# Accuracy del modelo
print("Accuracy del modelo en datos test:", accuracy_score(y_test, y_pred))

#Heat map de los datos obtenidos contra los reales
cm_df = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_df, annot=True, cmap='Blues')

#Matriz de confusion de los datos
cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred), columns = ['Yes','No'])
cm_df[''] = ['Yes', 'No']
cols = cm_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
cm_df = cm_df[cols]
cm_df = cm_df.set_index('')
print('Confusion matrix: \n')
print(cm_df)

