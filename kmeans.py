import pandas as pd
# util para poco recuersos 
from sklearn.cluster import MiniBatchKMeans
#funcion principal
if __name__ == "__main__":
    #lectura de set de datos de caramelos
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))
    #guardar set de datos sin el nombre del caramelo
    X = dataset.drop('competitorname', axis=1)
    #entrenar modelo 
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("Total de centros: " , len(kmeans.cluster_centers_))
    print("="*64)
    #print(kmeans.predict(X))
    #guardar la predicion en una columna de dataset
    dataset['group'] = kmeans.predict(X)
    

