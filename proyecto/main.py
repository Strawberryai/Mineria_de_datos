
# IMPORTS
from dockembeddings import train_docModel, load_docModel, vec_docEmbeddings
from hierarchical_clustering import hierarchical_clustering

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Para realizar comparaciones con la implementacion de scipy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, minkowski
from sklearn.metrics import silhouette_score

# VARIABLES GLOBALES
data_dir = "."
doc2vec_model_file = 'my_doc2vec_n50.model'
train_file = 'verbalAutopsy_train.csv'
test_file = 'verbalAutopsy_test.csv'

# MÉTODOS PARA SIMPLIFICAR
def train_hc_model(vectores, tipo_distancia, grado_minkowski=2):
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

def scatter_2D(vectores, labels, y_train):
    # Reducir las dimensiones para visualizarlas: PCA -> 2D
    pca = PCA(n_components=2)
    pca.fit(vectores)
    # Cambio de base a dos dimensiones PCA 
    x_train_PCAspace = pca.transform(vectores)
    print('Dimensiones después de aplicar PCA: ')
    samples = 300 # Número de instancias a dibujar
    # Dibujar los puntos en el espacio, color: cluster, etiqueta-numérica: clase
    # Color del punto: cluster
    sc = plt.scatter(x_train_PCAspace[:samples,0],x_train_PCAspace[:samples,1], cmap=plt.cm.get_cmap('nipy_spectral', 10),c=list(labels.values())[:samples])
    plt.colorbar()

    # Etiqueta numérica: clase 
    for i in range(samples):
        plt.text(x_train_PCAspace[i,0],x_train_PCAspace[i,1], y_train[i])
    plt.show()

def scatter_3D(vectores, labels):
    # Reducir las dimensiones para visualizarlas: PCA -> 3D
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

def calcular_silueta_para(lista_num_clusters, proc, dist_max=0):
    resultados = []
    for n in lista_num_clusters:
        sil = proc.silhouette_score(n, dist_max)
        resultados.append((n, sil))
        
    return resultados

def graficar_siluetas(siluetas):
        # Representamos los resultados en una gráfica de barras
        fig, ax = plt.subplots(figsize=(5,2))
        ax.set_xlabel('num clusters')
        ax.set_ylabel('Silhouette score')

        x = [silueta[0] for silueta in siluetas]
        y = [silueta[1] for silueta in siluetas]
        plt.bar(x, y, align='center', color='#007acc')
        plt.show()

def graficar_siluetas(siluetas1, siluetas2):
        # Representamos los resultados en un gráfico conjunto

        x1 = [silueta[0] for silueta in siluetas1]
        y1 = [silueta[1] for silueta in siluetas1]
        
        x2 = [silueta[0] for silueta in siluetas2]
        y2 = [silueta[1] for silueta in siluetas2]
        
        plt.plot(x1, y1, label='implementacion', color='blue')
        plt.plot(x2, y2, label='scipy', color='red')

        # Personalizar el gráfico
        plt.title('Comparación de siluetas')
        plt.xlabel('num clusters')
        plt.ylabel('Silhouette score')
        plt.legend()  # Muestra las etiquetas en el gráfico
        # Mostrar el gráfico
        plt.grid(True)
        plt.show()
        
# Algoritmo de scipy
def train_modelo_scipy(vectores):      
    distances = pdist(vectores, metric='minkowski', p=7.5)
    return linkage(distances, method='complete')

def calcular_silueta_scipy_para(lista_num_clusters, Z, vectores, criterion='maxclust'):
    resultados = []
    for n in lista_num_clusters:
        labels = fcluster(Z, t=n, criterion=criterion)
        sil = silhouette_score(vectores, labels, metric='euclidean')
        resultados.append((n, sil))
        
    return resultados 
    
    
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
    #vectores = x_train
    #vectores = vectores[0:2999] # 3000 instancias para el entrenamiento
    
    #proc = train_hc_model(vectores, 'complete', 2)
    
    proc = load_hc_model("complete_4_n50_3000.pkl")
    
    # Dibujar arbol completo
    #proc.draw_dendrogram()
    
    # Cortar el arbol en un numero determinado de clusters
    #proc.cortar_arbol(num_clusters=10,dist_max=0)
    
    # Obtener labels por cada vector -> diccionario
    #labels = proc.predict_multiple(vectores)
    #scatter_2D(vectores, labels, y_train)
    
    ## COMPROBAR DISTANCIAS DE NUEVOS DOCUMENTOS ##########
    nuevo_vector = list(vec_docEmbeddings(["Zakila isogailua sartu no en"], docModel))[0]
    print(nuevo_vector)
    proc.cortar_arbol(num_clusters=10,dist_max=0)
    
    # Obtener labels por cada vector -> diccionario
    labels = proc.predict(nuevo_vector)
    print(labels)
    
    
    ## SILUETA ##########
    # num_clusters = [i for i in range(2, 51)]
    # siluetas = calcular_silueta_para(num_clusters, proc)  
    # #graficar_siluetas(siluetas)
    # 
    # Z = train_modelo_scipy(vectores)
    # siluetas_scipy = calcular_silueta_scipy_para(num_clusters, Z, vectores)
    # #graficar_siluetas(siluetas_scipy)
    # 
    # graficar_siluetas(siluetas, siluetas_scipy)

    
if __name__ == "__main__":
    main()
    exit(0)
