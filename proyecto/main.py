
# IMPORTS
from dockembeddings import train_docModel, load_docModel, vec_docEmbeddings
from hierarchical_clustering import hierarchical_clustering

import pandas as pd

# VARIABLES GLOBALES
data_dir = "."
model_file = 'my_doc2vec.model'
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
    return hierarchical_clustering.load("complete_3.pkl")


# MAIN
def main():
    train = pd.read_csv(train_file)
    x_train = train['open_response']
    y_train = train['gs_text34']
    
    print(x_train.head())
    print(y_train.head())
    
    # Obtenemos la vectorizacion de los documentos -> [(index, vector)]
    # train_docModel(x_train, model_file)
    docModel = load_docModel(model_file)
    x_train = list(vec_docEmbeddings(x_train, docModel))
    vectores = [vec[1] for vec in x_train]
    
    proc = train_hc_model(vectores, 'complete', 7.5)
    # Dibujar arbol completo
    proc.draw_dendrogram()
    
    # Cortar el arbol en un numero determinado de clusters
    proc.cortar_arbol(num_clusters=3,dist_max=0)
    
    # Obtener labels por cada vector
    labels = proc.predict_multiple(vectores)
    print(labels)
    
    #proc = load_hc_model("complete_3.pkl")  
    #proc.cortar_arbol(num_clusters=4,dist_max=0)

if __name__ == "__main__":
    main()
    exit(0)
