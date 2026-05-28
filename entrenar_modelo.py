import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Leer dataset
dataframe = pd.read_csv("LIMPIO1.csv")

# Variables
X = dataframe.drop('Overall_Impact', axis=1)
y = dataframe['Overall_Impact']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear modelo KNN
modelo = KNeighborsClassifier(n_neighbors=3)

# Entrenar modelo
modelo.fit(X_train, y_train)

# Guardar modelo
joblib.dump(modelo, 'modelo_knn.pkl')

print("Modelo entrenado correctamente")