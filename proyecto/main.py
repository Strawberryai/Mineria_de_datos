
# IMPORTS
from dockembeddings import train_docModel, load_docModel, vec_docEmbeddings
from hierarchical_clustering import hierarchical_clustering

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# VARIABLES GLOBALES
data_dir = "."
doc2vec_model_file = 'my_doc2vec_n200.model'
train_file = 'verbalAutopsy_train.csv'
test_file = 'verbalAutopsy_test.csv'

# MÉTODOS PARA SIMPLIFICAR
def train_hc_model(vectores, tipo_distancia, grado_minkowski):
    """
    Método para entrenar un modelo de clustering jerárquico -> procesarCluster
    ...
    Parámetros
    ----------
    vectores : vec
        vector de los atributos de cada uno de los documentos
    tipo_distancia : str
        tipo de distancia intercluster para el entrenar el modelo: ['complete','single','mean','average']
    grado_minkowski : float
        grado de la distancia minkowski a utilizar
    """

    hc = hierarchical_clustering(vectores, tipo_distancia, p=grado_minkowski)
    proc = hc.cluster() # fit()
    proc.save() # Guardar el modelo
    
    return proc

def load_hc_model(filename):
    """
    Método para cargar un modelo de clustering jerárquico -> procesarCluster
    ...
    Parámetros
    ----------
    filename : str
        path del modelo a cargar
    """
    return hierarchical_clustering.load(filename)


# MAIN
def main():
    train = pd.read_csv(train_file)
    # Realizamos un encoding de las clases de las instancias: Pneumonia -> 0; Stroke -> 1...
    le = LabelEncoder()
    train['Clases'] = le.fit_transform(train['gs_text34']) 

    
    x_train = train['open_response']
    y_train = train['Clases']
    
    print(x_train.head())
    print(y_train.head())
    
    # Obtenemos la vectorizacion de los documentos -> [(index, vector)]
    # Entrenamos el modelo
    # train_docModel(pd.read_csv(train_file)['open_response'], model_file)
    docModel = load_docModel(doc2vec_model_file)
    x_train = list(vec_docEmbeddings(x_train, docModel))
    vectores = [vec[1] for vec in x_train]
    vectores = vectores[0:2999] # 3000 instancias para el entrenamiento
    
    #proc = train_hc_model(vectores, 'complete', 7.5)
    proc = load_hc_model("complete_4_n200.pkl")
    
    # Dibujar arbol completo
    #proc.draw_dendrogram()
    
    # Cortar el arbol en un numero determinado de clusters
    proc.cortar_arbol(num_clusters=10,dist_max=0)
    
    # Obtener labels por cada vector -> diccionario
    labels = proc.predict_multiple(vectores)
    #print(labels)

    ## Reducir las dimensiones para visualizarlas: PCA -> 2D
    #pca = PCA(n_components=2)
    #pca.fit(vectores)
    ## Cambio de base a dos dimensiones PCA 
    #x_train_PCAspace = pca.transform(vectores)
    #print('Dimensiones después de aplicar PCA: ')
    #samples = 300 # Número de instancias a dibujar
    ## Dibujar los puntos en el espacio, color: cluster, etiqueta-numérica: clase
    ## Color del punto: cluster
    #sc = plt.scatter(x_train_PCAspace[:samples,0],x_train_PCAspace[:samples,1], cmap=plt.cm.get_cmap('nipy_spectral', 10),c=list(labels.values())[:samples])
    #plt.colorbar()
#
    ## Etiqueta numérica: clase 
    #for i in range(samples):
    #    plt.text(x_train_PCAspace[i,0],x_train_PCAspace[i,1], y_train[i])
    #plt.show()
    
    
    # Reducir las dimensiones para visualizarlas: PCA -> 2D
    pca = PCA(n_components=3)
    pca.fit(vectores)
    # Cambio de base a dos dimensiones PCA 
    x_train_PCAspace = pca.transform(vectores)
    print('Dimensiones después de aplicar PCA: ')
    samples = 300 # Número de instancias a dibujar
    # Dibujar los puntos en el espacio, color: cluster, etiqueta-numérica: clase
    # Color del punto: cluster
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train_PCAspace[:samples,0],x_train_PCAspace[:samples,1], x_train_PCAspace[:samples,2], cmap=plt.cm.get_cmap('nipy_spectral', 10),c=list(labels.values())[:samples])
    ax.set_xlabel('Eje X')
    ax.set_xlabel('Eje Y')
    ax.set_xlabel('Eje Z')
    plt.show()
    
if __name__ == "__main__":
    main()
    exit(0)
