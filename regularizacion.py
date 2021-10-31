import pandas as pd
import sklearn
#librerias de los modelos
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
#dividir nuestros datos entre conjunto de entranimiento y test
from sklearn.model_selection import train_test_split
#metricas del error medio cuadrado
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    #cargar data set
    dataset = pd.read_csv('./data/whr2017.csv')
    print(dataset.describe())
    X = dataset[['gdp', 'family', 'lifexp', 'freedom' , 'corruption' , 'generosity', 'dystopia']]
    y = dataset[['score']]
    #dimension de los set de datos
    print(X.shape)
    print(y.shape)
    #separar nuestros datos para el model
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
    #implementar modelo de regresion lineal
    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear =  modelLinear.predict(X_test)
    #implementacion de la regularizacion tipo Lasso
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)
    #implementacion de la regularizacion tipo Ridge
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)
    # lectura de la metrica del error cuadrado desde un modelo lineal
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Perdida desde el modelo lineal: ", linear_loss)
    # lectura de la metrica del error cuadrado aplicando regularizacion Lasso
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Perdida con regresion tipo Lasso: ", lasso_loss)
    # lectura de la metrica del error cuadrado aplicando regularizacion Ridge
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Perdida con regresion tipo Ridge: ", ridge_loss)
    #comparacion de efectividad segundo sus coeficientes
    print("="*32)
    #los coeficientes son el numero de columnas del set de datos y si estos son mayores siginifica que esa columna tiene mucha mas peso en la prediccion.
    print("Coeficientes de perdida LASSO: ",modelLasso.coef_)
    print("="*32) 
    print("Coeficientes de perdida  RIDGE",modelRidge.coef_)

#implementacion_lasso_ridge