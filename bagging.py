import pandas as pd
#clasificador modelo vecino mas cercano
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
#herramientas para separa set de datos y metrica de efectividad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    #lectura de datos
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())
    #borrar columna de predcion
    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']
    #separar set de datos para entrenar el modelo 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35)
    #llamar modelo de vecino mas cercano 
    knn_class = KNeighborsClassifier().fit(X_train,y_train)
    knn_pred = knn_class.predict(X_test)
    #compracion de modelos vs bagging
    print("="*64)
    #precision de vecino mas cercano 
    print(accuracy_score(knn_pred, y_test))    
    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(),n_estimators=10).fit(X_train,y_train)
    bag_pred = bag_class.predict(X_test)
    #precision de bagging
    print("="*64)
    print(accuracy_score(bag_pred, y_test))  