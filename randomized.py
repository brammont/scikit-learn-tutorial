import pandas as pd
#herrmient de optimizacion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
#funcion principal
if __name__ == "__main__":
    #leer data ser
    dataset = pd.read_csv('./data/felicidad.csv')
    #borrar columnas innecesarias
    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset[['score']]
    #entrenar modelo de optimizacion
    reg = RandomForestRegressor()
    #parametros del dicionario para optener la mejor optimizacion de tipo aletorea
    parametros = {
        'n_estimators' : range(4,16),
        'criterion' : ['mse', 'mae'],
        'max_depth' : range(2,11)
    }
    #instancia de la optimizacion
    rand_est = RandomizedSearchCV(reg, parametros , n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)
    #valores de los clusters
    print("mejor estimador: ",rand_est.best_estimator_)
    print("parametors configuracion: ",rand_est.best_params_)
    print("valores de prediccion de la primera fila: ",rand_est.predict(X.loc[[0]]))

    #implmentacion_randomizedSearchCV