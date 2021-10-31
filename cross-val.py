import pandas as pd
import numpy as np
#modelo de arbol de desicon
from sklearn.tree import DecisionTreeRegressor
#algortimo de validacion cruzudad y de tipo k.folds
from sklearn.model_selection import (
    cross_val_score, KFold
)
#funcion principal
if __name__ == "__main__":
    #lectura de set de datos
    dataset = pd.read_csv('./data/felicidad.csv')
    #borrar columna de pais y la cual se va a validar
    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']
    #entreanar modelo de desicion
    model = DecisionTreeRegressor()
    #calculo de metrica de validacion cruzada
    score = cross_val_score(model, X,y, cv= 3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))
    #validacion cruzada de tipo k-folds
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train)
        print(test)

    #implementacion_cross_validation
