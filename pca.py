import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart.head(5))
    dt_features  = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']
    dt_features = StandardScaler().fit_transform(dt_features)
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
    #imprimir si son del sima dimension
    print('Demensiones:\nSet de datos{}\nSet de resultados{}'.format(X_train.shape,y_train.shape))
    #n_componentes = min(n_muestras, n_featurnes)
    pca = PCA(n_components=3)
    #entranr el pca
    pca.fit(X_train)
    #alternativa con bajo redimiento lo divide dependiendo el bunero de bloues o bach
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    #representacion grafica de la varianza de los sets
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()
    #comparar los dos algortimos
    logistic = LogisticRegression(solver='lbfgs')
    #llamar el algortimo desde el conjunto de entrenamiento
    dt_train = pca.transform(X_train)
    #subconjunto de prueba
    dt_test = pca.transform(X_test)
    #entrenar el model
    logistic.fit(dt_train, y_train)
    #mostrar metrica
    print("SCORE PCA: ", logistic.score(dt_test, y_test))
    #Ahora desde el otro algortimo IPCA
    dt_train = ipca.transform(X_train)
    #subconjunto de prueba
    dt_test = ipca.transform(X_test)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))
    

