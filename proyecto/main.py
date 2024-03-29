
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


import time
import numpy as np
# VARIABLES GLOBALES
data_dir = "."
doc2vec_model_file = 'my_doc2vec_n100.model'
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
        "4":    ["Stroke", "Acute Myocardial Infarction"],
        "5":    ["Pneumonia", "Asthma", "COPD"],
        "6":    ["Cirrhosis", "Other Digestive Diseases"],
        "7":    ["Renal Failure"],
        "8":    ["Preterm Delivery", "Stillbirth", "Maternal", "Birth Asphyxia"],
        "9":    ["Congenital Malformations"],
        "10":   ["Bite of Venomous Animal", "Poisonings"],
        "11":   ["Road Traffic", "Falls", "Homicide", "Fires", "Drowning", "Suicide", "Violent Death", "Other injuries"]
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

def obtener_train_mas_cercano(documentos, train, proc, docModel):
    test_vectors    = list(vec_docEmbeddings(documentos, docModel))
    test_labels     = []
    test_dists      = []
    textos_originales   = []
    textos_cercanos     = []
    
    for i, vec in enumerate(test_vectors):
        label, dist, index = proc.obtener_indice_instancia_mas_cercana(vec)
        test_labels.append(i)
        test_dists.append(dist)
        
        texto_original = documentos[i]
        texto_cercano = train.iloc[index]['open_response']
        textos_originales.append(texto_original)
        textos_cercanos.append(texto_cercano)
        
        print()
        print(f"TEXTO ORIGINAL TEST dist: {dist}")
        print(texto_original)
        print(f"TEXTO CERCANO TRAIN index: {index}")
        print(texto_cercano)
        print()
        
    graficar_distancias_documentos(test_labels, test_dists, textos_originales, textos_cercanos)

def graficar_distancias_documentos(x, y, texts1, texts2):   
    # Función para acortar el texto y agregar puntos suspensivos
    def acortar_texto(texto, longitud_maxima=70):
        if len(texto) > longitud_maxima:
            return texto[:longitud_maxima - 3] + '...'
        else:
            return texto
    
    # Asigna colores en función de los valores
    colores = plt.cm.viridis(np.array(y) / max(y))

    # Crea el gráfico de barras horizontales
    plt.barh(x, y, color=colores)

    # Agrega texto a cada barra
    for i, y in enumerate(y):
        t1 = acortar_texto(texts1[i])
        t2 = acortar_texto(texts2[i])
        plt.text(0, i, f'{t1}\n{t2}', color='black', va='center')

    # Personaliza el gráfico
    plt.ylabel('Indice del texto del test')
    plt.xlabel('Distancia al cluster más cercano')
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
    samples = 300 # NúmlogisticRegr.fit(x_train, y_train)ero de instancias a dibujar
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
    train['gs_text34'] = train['gs_text34'].str.lower()
    # Realizamos un encoding de las clases de las instancias en función de un mapeo definido: Pneumonia -> 0; Stroke -> 1...
    train = mapeo_de_labels(train, 'gs_text34') # Añade la columna 'Clases'
    x_train = train['open_response']
    y_train = train['Clases']
    
    # print(x_train.head())
    # print(y_train.head())
    
    # Obtenemos la vectorizacion de los documentos -> [(index, vector)]
    # Entrenamos el modelo
    #train_docModel(pd.read_csv(train_file)['open_response'], doc2vec_model_file)
    docModel = load_docModel(doc2vec_model_file)
    x_train = list(vec_docEmbeddings(x_train, docModel))
    vectores = x_train
    vectores = vectores[0:999] # 1000 instancias para el entrenamiento
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
    
    #proc = train_hc_model(vectores, 'complete', 2)
    
    proc = load_hc_model("complete_4_n100_1000_2grado.pkl")
    
            
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
    proc.cortar_arbol(num_clusters=10,dist_max=0)
    proc.cortar_arbol(num_clusters=8,dist_max=0)
    
    # Obtener labels por cada vector
    labels = proc.predict_multiple(vectores)

    
    ## PRUEBAS
    ## N ÓPTIMO Doc2Vec #################################################
    #calcular_n_optimo_doc2vec()
    
    ## DISTANCIAS A DOCUMENTOS ##########################################
    test = [
        "the caese of death is pneumonia",
        "the deceased had fallen off a bike and died",
        "the deceased had severe anemia",
        "the deceased was killed by a sharp weapon",
        "aaa aaab aba abbba bbbbbb",
        "o xd"
    ]
    obtener_train_mas_cercano(test, train, proc, docModel)
       
    
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
