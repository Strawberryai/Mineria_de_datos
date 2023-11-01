# IMPORTS
from dockembeddings import train_docModel, load_docModel, vec_docEmbeddings
from hierarchical_clustering import hierarchical_clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Para realizar comparaciones con la implementacion de scipy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, minkowski
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from scipy.optimize import linear_sum_assignment

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

def modificar_etiquetas(labels):
    minimo = min(labels)
    maximo = max(labels)
    min_original = 0
    max_original = 47
   
    dict_mapeo = {}
    for i in range(minimo,maximo):
        dict_mapeo[min_original]= i 
        min_original += 1
    
    print(dict_mapeo)
    input("jodienda")
    etiquetas_mapeadas = [dict_mapeo[etiqueta] for etiqueta in labels]
    
    print(etiquetas_mapeadas)
    input("jodienda")
    return etiquetas_mapeadas
    
    valores_mapeados = min_original + ((labels - minimo) / amplitud) * (max_original - min_original)
    return valores_mapeados
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

def graficar_cofonetica(c_datos):
    plt.hist(c_datos, bins=10, color='blue', edgecolor='black')
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de Valores')
    plt.show()

def graficar_dimensionalidad(proc):
    def graph(ax, pca_, color, n95, n99):
        PC_values = np.arange(pca_.n_components_) + 1
        ax.plot(PC_values, pca_.explained_variance_ratio_, 'o-', linewidth=2, color=color)
        
        # Marcar valores n95 y n99 en el eje x con líneas punteadas
        ax.axvline(x=n95, color="orange", linestyle="--", label="95%")
        ax.axvline(x=n99, color="red", linestyle="--", label="99%")
        
        # Agregar una leyenda
        ax.legend()
        
        plt.show()
        
    proc.cortar_arbol(num_clusters=8, dist_max=0)
    dim = max(len(vec) for vec in proc.vectors)
    print(dim)
    #dim = 500
    pca_ = PCA(n_components=dim)
    pca_.fit(proc.vectors)
    print(pca_.n_components_)
    pca__= PCA()
    pca__.fit(proc.vectors)

    # Calcula la varianza explicada acumulada
    cumulative_variance_ratio = np.cumsum(pca__.explained_variance_ratio_)
    # Encuentra el número de componentes que preservan al menos el 95% de la varianza
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    # Encuentra el número de componentes que preservan al menos el 99% de la varianza
    n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
    
    # simplemente graficar
    fig, (ax1) = plt.subplots(1)
    fig.suptitle('Variación de datos según la dimension')
    
    graph(ax1, pca_, color='blue', n95=n_components_95, n99=n_components_99)
    print(dim)
    print(n_components_95)
    print(n_components_99)
    
    return (dim, n_components_95, n_components_99)


def evaluacion_de_metricas(num_clusters, proc, X_train, y_train, cm, true_labels, pred_labels):
    graficar_dimensionalidad(proc)
    input("dimensionalidad")
    proc.metrics_evaluation()
    
def class_to_cluster(proc, x_train, y_train):
    true_labels = y_train[0:len(x_train)]
    print(true_labels)
    input("t_labels")
    dicc_t_labels = {}
    lista_nones = []
    lista_t_labels = []
    # Creando diccionario de los labels verdaderos
    for i in range(0, len(x_train)):
        nuevo_valor = true_labels[i]
        if nuevo_valor is not None:
            dicc_t_labels[i] = nuevo_valor
            lista_t_labels.append(i)
        else:
            lista_nones.append(i)
        
    print(lista_nones)
    lista_t_labels = [int(valor) for valor in dicc_t_labels.values()]
    print(lista_t_labels)
    input("")
    print("Etiquetas reales: ", len(lista_t_labels))
    print("Instancias sin etiquetas reales: ", len(lista_nones))
    input("recuento")
    t_values, t_counts = np.unique(lista_t_labels, return_counts=True)
    print(t_values)
    print(t_counts)
    #[ 0  1  2  3  4  5  6  7  8  9 11 12]
    # [532 146 101  10 193 119 353  96 115 524  64 397]
    input("values")
    
    t_labels_values_ordenados = np.array(t_values)[np.argsort(-np.array(t_counts))].tolist()
    print(t_labels_values_ordenados)
    input("ordenacion")
    #[0, 9, 12, 6, 4, 1, 5, 8, 2, 7, 11, 3]
    
    
    proc.cortar_arbol(num_clusters=len(t_values),dist_max=0)
    labels = proc.predict_multiple(proc.vectors)
    print(labels)
    
    p_labels_values, p_counts = np.unique(list(labels.values()), return_counts=True)
    print(p_labels_values)
    print(p_counts)
    print(proc.clusters)
    input("labels")
    #[5968 5969 5970 5972 5974 5975 5980 5981 5982 5983 5984 5985]
    #[ 611    9   23   58   29  129  202  171   49  143 1459  116]
    # Ordenar labels de los predichos para que coincidan con los verdaderos
    
    p_labels_values_ordenados = np.array(p_labels_values)[np.argsort(-np.array(p_counts))].tolist()
    print(p_labels_values_ordenados)
    input("ordenacion values predichos")
    # [5984, 5968, 5980, 5981, 5983, 5975, 5985, 5972, 5982, 5974, 5970, 5969]
    
    lista_p_labels = []
    lista_p_nones = []
    # crear lista de los labels predichos, sin nones
    # escalado a chapters
    for idx in range(0, len(labels)):
        # obtenemos la etiqueta predicha de cada vector
        nuevo_valor = labels[idx]
        if idx in lista_nones:
            lista_p_nones.append(idx)
        else:
            # guardamos su etiqueta real correspondiente
            i = p_labels_values_ordenados.index(nuevo_valor) 
            #i = proc.clusters.index(nuevo_valor)
            lista_p_labels.append(i)
            
    print(lista_p_labels)
    print(lista_t_labels)
    input("confusion matrix")
    
    to_string = lambda x : str(x)
    # Obtener matriz de confusión Class to clustering eval
    #cm = confusion_matrix(np.vectorize(to_string)(lista_p_labels), np.vectorize(to_string)(lista_t_labels))
    cm = confusion_matrix(lista_t_labels, lista_p_labels)
    
    """
    cm_disp = ConfusionMatrixDisplay(cm)
    cm_disp.plot()
    plt.show()
    """
    # Mapa de calor a partir de la matriz de confusion    
    ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

    plt.show()

    
    row_ind, col_ind = linear_sum_assignment(-cm)
    new_cm = cm[:, col_ind]
    

    # Crear una nueva matriz de confusión reasignando las columnas
    #cm_nuevo = np.array([cm[:, mapeo[label]] for label in values])
    
    ax = sns.heatmap(new_cm, annot=True, cmap="Blues", fmt="d")
    ax.set_xlabel("Predicción (cluster)")
    ax.set_ylabel("Verdadero")
        
    plt.show()
    
    total=0
    for i in range(len(new_cm)):
        sumaTot=0
        for j in range(len(new_cm)):
            sumaTot=new_cm[j][i]+sumaTot
        error=1-(new_cm[i][i])/sumaTot
        print("El error del cluster "+str(i)+" es de:"+str(error))
        total=total+error
    error_tot=total/(len(new_cm))
    print("El error promedio es:"+str(error_tot))
    
    return new_cm, lista_t_labels, lista_p_labels

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
    print(train)
    y_train = train['Clases']
    print(y_train.unique())
    plt.bar(y_train.value_counts().index, y_train.value_counts())
    plt.show()
    input("valores unicos")
    
    #print(x_train.head())
    #print(y_train.head())
    
    model_file = 'doc2model.model'
    # Obtenemos la vectorizacion de los documentos -> [(index, vector)]
    # Entrenamos el modelo
    #train_docModel(pd.read_csv(train_file)['open_response'], model_file)
    
    input("primer train docModel")
    docModel = load_docModel(model_file)
    x_train = list(vec_docEmbeddings(x_train, docModel))
    vectores = x_train
    vectores = vectores[0:1999] # 2000 instancias para el entrenamiento
    
    #proc = train_hc_model(vectores, 'complete', 2)
    input("cargar modelo hc")
    proc = load_hc_model("complete_4_2000.pkl") # <---- n50, 2000
    """
    proc.cortar_arbol(num_clusters=13,dist_max=0)
    print(proc.clusters)
    print(len(proc.clusters))
    input("clusts")
    #proc.dist_cofonetica()
    #proc.graficacion_siluetas()
    """

    #proc.cortar_arbol(num_clusters=10,dist_max=0)
    """
    labels = proc.predict_multiple(proc.vectors)
    print(labels)
    print(len(labels))
    print(min(labels), max(labels))
    values, counts = np.unique(labels, return_counts=True)
    print(values)
    print(counts)
    """
    #print(y_train[0:2999])
    input("class cluster")
    cm, t_labels, p_labels = class_to_cluster(proc, vectores, y_train)
    # Dibujar arbol completo
    #proc.draw_dendrogram()
    input("cofonetica")
    c_datos = proc.dist_cofonetica()
    input("dims")
    graficar_dimensionalidad(proc)
    evaluacion_de_metricas(num_clusters=12, proc=proc, X_train=x_train, y_train=y_train, cm=cm, true_labels=t_labels, pred_labels=p_labels)
    
    
    # Cortar el arbol en un numero determinado de clusters
    #proc.cortar_arbol(num_clusters=10,dist_max=0)
    labels = proc.predict_multiple(vectores)
    
    ## COMPROBAR DISTANCIAS DE NUEVOS DOCUMENTOS ##########
    nuevo_vector = list(vec_docEmbeddings(["Zakila isogailua sartu no en"], docModel))[0]
    print(nuevo_vector)
    proc.cortar_arbol(num_clusters=10,dist_max=0)
    
    # Obtener labels por cada vector -> diccionario
    labels = proc.predict(nuevo_vector)
    print(labels)
    input("labels")
    
    
    
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
