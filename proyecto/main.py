
# IMPORTS
from dockembeddings import train_docModel, load_docModel, vec_docEmbeddings
from hierarchical_clustering import hierarchical_clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import time
import numpy as np
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
    x_train = train['open_response']
    y_train = train['gs_text34']
    
    print(x_train.head())
    print(y_train.head())
    
    # Obtenemos la vectorizacion de los documentos -> [(index, vector)]
    # Entrenamos el modelo
    # train_docModel(pd.read_csv(train_file)['open_response'], model_file)
    docModel = load_docModel(doc2vec_model_file)
    x_train = list(vec_docEmbeddings(x_train, docModel))
    vectores = [vec[1] for vec in x_train]
    vectores_predict=vectores[3000:3200]
    vectores = vectores[0:2999] # 3000 instancias para el entrenamiento
    vectores_predict
    import csv
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, minkowski
    import psutil
    import os
# Abre un archivo CSV para escribir los tiempos
    results=[]
    with open('tiempos.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Tamaño", "Tiempo Bloque 1", "Tiempo Bloque 2"])  # Escribir encabezados
        for x in [25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 700, 1000, 1500,2000,3000]:
            vectors = vectores[0:x]
            """
            proc = hierarchical_clustering(vectors, 'complete', p=2)
            proc=proc.cluster()
            proc.save(x)
            #proc.draw_dendrogram()
            
            end_time = time.time()
            elapsed_time_block1 = end_time - start_time
            # Medir el uso de memoria RAM después del primer bloque
            memory_usage_block1 = psutil.Process(os.getpid()).memory_info().rss
            # Aquí se ejecuta el segundo bloque de código
            """
            data = vectors  # Asigna 'vectors' a 'data' para el segundo bloque
            distances = pdist(data, metric='minkowski', p=2)
            Z = linkage(distances, method='complete')
            dendrogram(Z)
    
            
            # Escribir los tiempos en el archivo CSV
            #writer.writerow([x, elapsed_time_block1])
            memory_usage_block1 = psutil.Process(os.getpid()).memory_info().rss
            results.append([x, memory_usage_block1])

    with open('memory_usage_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "Memory Usage (Block 2)"])
        writer.writerows(results)
    # Dibujar arbol completo
    proc.draw_dendrogram()
    
    # Cortar el arbol en un numero determinado de clusters
    proc.cortar_arbol(num_clusters=8,dist_max=0)
    
    # Obtener labels por cada vector -> diccionario
    labels = proc.predict_multiple(vectores)
    #print(labels)

    

    # Distancia de Minkowski con p=7.5
   

    # Datos de ejemplo
    data = np.array(vectores)

    # Parámetros
    num_clusters =8
    p = 7.5  # Parámetro para la distancia de Minkowski

    # Calcula las distancias de Minkowski entre los puntos
    distances = pdist(data, metric='minkowski', p=p)

    # Realiza el agrupamiento jerárquico con distancia completa
    Z = linkage(distances, method='complete')

    # Genera el dendrograma
    dendrogram(Z)

    # Muestra el dendrograma
    plt.show()

    # Realiza la asignación de clústeres a los datos de ejemplo
    clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

   # Realiza la asignación de clústeres a los datos de ejemplo
    clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

    # Imprime las etiquetas de clústeres para cada vector
    recuento_clusters_nuestro={}
    recuento_clusters_real={}
    for x in proc.clusters:
        recuento_clusters_nuestro[x]=len(proc.obtener_nodos_finales(x))
    for i, cluster_label in enumerate(clusters):
        if(i>2998):
            print (recuento_clusters) 
            break
        try:        
            if cluster_label in recuento_clusters_real:
                recuento_clusters_real[cluster_label] += 1
            else:
                recuento_clusters_real[cluster_label] = 1
        except casoFuera:
            break
    print(recuento_clusters_nuestro)
    print (recuento_clusters_real)        
        

# Imprime la asignación de clústeres de los vectores de predicción
    
    """
    # Reducir las dimensiones para visualizarlas: PCA
    pca = PCA(n_components=2)
    pca.fit(vectores)
    # Cambio de base a dos dimensiones PCA 
    x_train_PCAspace = pca.transform(vectores)
    print('Dimensiones después de aplicar PCA: ',x_train_PCAspace.shape)  
    samples = 300 # Número de instancias a dibujar
    # Dibujar los puntos en el espacio, color: cluster, etiqueta-numérica: clase
    # Color del punto: cluster
    sc = plt.scatter(x_train_PCAspace[:samples,0],x_train_PCAspace[:samples,1], cmap=plt.cm.get_cmap('nipy_spectral', 10),c=list(labels.values())[:samples])
    plt.colorbar()
    
    # Etiqueta numérica: clase 
    #for i in range(samples):
        #plt.text(x_train_PCAspace[i,0],x_train_PCAspace[i,1], y_train[i])
    plt.show()
    """
if __name__ == "__main__":
    main()
    exit(0)
