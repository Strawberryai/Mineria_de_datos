
# IMPORTS
from dockembeddings import train_docModel, load_docModel, vec_docEmbeddings
from hierarchical_clustering import hierarchical_clustering

import pandas as pd
import numpy as np
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
def mapeo_de_labels(datos_df, columna):
    # PRE: dataframe con todos los datos. "columna" es la entrada que contiene los labels reales
    # POST: devuelve el mismo dataframe añadiendo la columna "Clases" en las cuales se mapean los valores de la
    #       columna que se recibe como parámetro en funcion del diccionario "mapeo"
    mapeo = {
        "0":    ["Diarrhea/Dysentery", "Other infectious diseases", "AIDS", "Sepsis", "Meningitis", "Meningitis/Sepsis", "Malaria", "Encephalitis", "Measles", "Hemorrhagic Fever", "TB"],
        "1":    ["Leukemia/Lymphomas", "Colorectal Cancer", "Lung Cancer", "Cervical Cancer", "Breast Cancer", "Stomach Cancer", "Prostate Cancer", "Esophageal Cancer", "Other Cancers"],
        "2":    ["Diabetes"],
        "3":    ["Epilepsy"],
        "4":    ["Stroke"],
        "5":    ["Acute Myocardial Infarction"],
        "6":    ["Pneumonia", "Asthma", "COPD"],
        "7":    ["Cirrhosis", "Other Digestive Diseases"],
        "8":    ["Renal Failure"],
        "9":    ["Preterm Delivery", "Stillbirth", "Maternal", "Birth Asphyxia"],
        "10":   ["Congenital Malformations"],
        "11":   ["Bite of Venomous Animal", "Poisonings"],
        "12":   ["Road Traffic", "Falls", "Homicide", "Fires", "Drowning", "Suicide", "Violent Death", "Other injuries"]
    }
    # Convertir el diccionario de mapeo original a minúsculas
    mapeo = {etiqueta.lower(): [clase.lower() for clase in clases] for etiqueta, clases in mapeo.items()}

    # Función personalizada para mapear
    def mapear_etiqueta(clase):
        for etiqueta, clases_lista in mapeo.items():
            if clase in clases_lista:
                return etiqueta
        return None

    datos_df['gs_text34'] = datos_df['gs_text34'].str.lower()
    datos_df['Clases'] = datos_df['gs_text34'].apply(mapear_etiqueta)
    
    return datos_df


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

def calcular_n_optimo_doc2vec():
    # Banco de pruebas -> 500 instancias de entrenamiento del train | distancia inter cluster: complete | distancia minkowski: 2 | de 2 a 15 como num_clusters | dist_max = 0
    train = pd.read_csv(train_file)['open_response']
    
    n_doc2vec = [50, 100, 150, 200, 250, 300]
    num_clusters = [i for i in range(2, 16)]
    datos = []
    for n in n_doc2vec:
        train_docModel(train, f"{n}.model")
        docModel = load_docModel(f"{n}.model")
        vectores = list(vec_docEmbeddings(train, docModel))[:499]
        
        proc = train_hc_model(vectores, 'complete', 2)      
        s = calcular_silueta_para(num_clusters, proc, dist_max=0)
        datos.append(s)
    
    
    # Graficamos los resultados
    colors = ['blue', 'red', 'green', 'black', 'magenta', 'yellow']
    for i, n in enumerate(n_doc2vec):
        siluetas = datos[i]
        x = [silueta[0] for silueta in siluetas]
        y = [silueta[1] for silueta in siluetas]

        plt.plot(x, y, label=f'N Doc2Vec = {n}', color=colors[i])

    # Personalizar el gráfico
    plt.title('Comparación de siluetas')
    plt.xlabel('num clusters')
    plt.ylabel('Silhouette score')
    plt.legend()  # Muestra las etiquetas en el gráfico
    # Mostrar el gráfico
    plt.grid(True)
    plt.show()  
    

def graficar_distancias_documentos(distancias):
    x = [d[0] for d in distancias]
    y = [d[1] for d in distancias]
    
    # Asigna colores en función de los valores
    colores = plt.cm.viridis(np.array(y) / max(y))
    
    # Crea el gráfico de barras
    plt.bar(x, y, color=colores)

    # Personaliza el gráfico
    plt.xlabel('Indice documento')
    plt.ylabel('Distancia al cluster más cercano')
    plt.title('Estudio de la distancia de documentos a sus respectivos clusters')
    plt.xticks(rotation=45)  # Rota las etiquetas del eje x para mayor legibilidad

    # Muestra el gráfico
    plt.show()

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
    datos_df['gs_text34'] = datos_df['gs_text34'].str.lower()
    # Realizamos un encoding de las clases de las instancias en función de un mapeo definido: Pneumonia -> 0; Stroke -> 1...
    train = mapeo_de_labels(train, 'gs_text34') # Añade la columna 'Clases'
    x_train = train['open_response']
    y_train = train['Clases']
    
    # print(x_train.head())
    # print(y_train.head())
    
    # Obtenemos la vectorizacion de los documentos -> [(index, vector)]
    # Entrenamos el modelo
    # train_docModel(pd.read_csv(train_file)['open_response'], model_file)
    docModel = load_docModel(doc2vec_model_file)
    x_train = list(vec_docEmbeddings(x_train, docModel))
    vectores = x_train
    vectores = vectores[0:2999] # 3000 instancias para el entrenamiento
    
    #proc = train_hc_model(vectores, 'complete', 2)
    
    proc = load_hc_model("complete_4_n50_3000.pkl")
    
    # Dibujar arbol completo
    #proc.draw_dendrogram()
    
    # Cortar el arbol en un numero determinado de clusters
    proc.cortar_arbol(num_clusters=10,dist_max=0)
    
    # Obtener labels por cada vector
    labels = proc.predict_multiple(vectores)
    
    
    ## PRUEBAS
    
    ## N ÓPTIMO Doc2Vec #################################################
    #calcular_n_optimo_doc2vec()
    
    ## DISTANCIAS A DOCUMENTOS ##########################################
    test = [
        "the deceased died after having high fever and anaemia",
        "the deceased was killed by a sharp weapon",
        "my daughter while playing slipped and fell down in the sump more water was there at that time hence she plunged in and died nobody has seen her felling in to that other wise this would have not happened",
        "aba bab bb aa aa bha",
        "child was absolutely fine he had no problem he was taken to hospital when he felt some problem after taking poison poison was given by father",
        "the baby died in the womb because mother had malaria",
        "ppppppppppp pppppppppppp pppppppppppp ppppppppppppp ppppppppppppppppppppppppppppppppppppppppppppp"
    ]
    
    test_vectors = list(vec_docEmbeddings(test, docModel))
    print(test_vectors)
    test_predict         = [proc.predict(x) for x in test_vectors] 
    test_labels     = [x[0] for x in test_predict]
    test_distancias = [(i, x[1]) for i,x in enumerate(test_predict)]
    print(test_distancias)
    graficar_distancias_documentos(test_distancias)
     
    
    ## SCATTER 2D y 3D ##################################################
    scatter_2D(vectores, labels, y_train)
    scatter_3D(vectores, labels)
    
    ## SILUETAS #########################################################
    num_clusters = [i for i in range(2, 51)]
    siluetas = calcular_silueta_para(num_clusters, proc)  
    #graficar_siluetas(siluetas)
    Z = train_modelo_scipy(vectores)
    siluetas_scipy = calcular_silueta_scipy_para(num_clusters, Z, vectores)
    #graficar_siluetas(siluetas_scipy)
    graficar_siluetas(siluetas, siluetas_scipy)

    
if __name__ == "__main__":
    main()
    exit(0)
