import pandas as pd
#llamar varias funciones de la libreria
from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
#modelo de mauinas de soporte vectorial especificamente el regresor
from sklearn.svm import SVR
#herrramientas para cargar los datos y medir metricas
from sklearn.model_selection import train_test_split
from sklearn.metrics  import mean_squared_error
#funcion princial
if __name__ == "__main__":
    #lectura de datos
    dataset =  pd.read_csv('./data/felicidad_corrupt.csv')
    print(dataset.head(5))
    #borrar columnas para limpieza
    X = dataset.drop(['country', 'score'], axis=1)
    #set de datos a predecir.
    y = dataset[['score']]
    #separar set de datos para entreno de modelo
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    #diccionario con la configuracion de los stimadores
    estimadores = {
        'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1), 
        'RANSAC' : RANSACRegressor(),#metaestimador
        'HUBER' : HuberRegressor(epsilon=1.35)
    }
    for name, estimator in estimadores.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        print("="*64)
        print(name)
        print("MSE: ", mean_squared_error(y_test, predictions))

#implementacion_regresion_robusta

