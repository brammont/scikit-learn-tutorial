import pandas as pd
import sklearn
#herramienta de algoritmo de boosting
from sklearn.ensemble import GradientBoostingClassifier
#herramientas para separar set de datos y calculo de metrica de presicion de los modelos ml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#funcion principal
if __name__ == "__main__":
    #leer set de datos
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())
    #borrar columna a predecir
    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']
    #separar set de datos
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    #instanciar algortimo de tipo arbol de desicion para motodos de ensamble.
    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    #presicion del modelo 
    print("="*64)
    print(accuracy_score(boost_pred, y_test))

#implementacion_boosting
