import pandas as pd
#herramienta de cloustering para pocos datos recordar
from sklearn.cluster import MeanShift
#funcion principal
if __name__ == "__main__":
    #lectura de set de datos
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(5))
    #borrar nombres de dulces
    X = dataset.drop('competitorname', axis=1)
    #entrenar modelo con parametros por defecto
    meanshift = MeanShift().fit(X)
    print("cuantos grupos encontro: ",max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print("="*64)
    print(dataset)

    
