from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import random
from scipy.spatial import distance
import pickle




# BORRAR EN FUTURO
#from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec
#import gensim.downloader
#import smart_open

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd

#import os
#from dockembeddings import vec_docEmbeddings
def help():
    print("########################Ayuda para el uso de la clase hierarchical_clustering####################################################")
    print("#1.-Cuando se llama a la constructora se le dan los vectores, el grado para la distancia minkowski y el tipo de distancia intergrupal")
    print("#    ejemplo : hc = hierarchical_clustering(vectors, distance_type,p=3)")
    print("#2.-Para iniciar la clusterización se llama al metodo cluster de esa misma clase")
    print("#    ejemplo : proc = hc.cluster()")
    print("#Este metodo devuelve un objeto de la clase procesarCluster, que como su nombre indica es la clase que se encarga de analizar el arbol generado por la clusterizacion")
    print("#3.-Si queremos, por ejemplo, obtener 4 clusters con una distancia maxima de 20 se podría expresar tal que así")
    print("#    ejemplo : proc.buscar_nodos(num_clusters=4,dist_max=0)")
    print("#4.-Esta clase también se encarga de hacer las predicciones, tiene dos metodos para esto, predict y predict_multiple, el primero devuelve el lavel de un vector, y el segundo devuelve el de varios vectores en un diccionario")
    print("#    ejemplo :  labels=proc.predict_multiple(test)")
    print("#En ese último caso se devolverian los labels del test que contendría un array de vectores")
    print("#5.-Finalmente, se podrá visualizar el arbol generado en la clase hierarchical clustering mediante el metodo draw_dendrogram()")
    print("#    ejemplo : hc.draw_dendrogram()")
    print("#################################################################################################################################")
class procesarCluster():
    def __init__(self,vectors,arbol,num_clusters=4,dist_max=20,distance_type='single',p=2):
        self.distance_type=distance_type
        self.vectors=vectors
        self.tree=arbol
        self.num_clusters=num_clusters
        self.dist_max=dist_max
        self.nodoPadre=max(self.tree.keys())
        self.clusters=[]
        self.grado=p
        self.centroides={}
    def obtener_nodos_finales(self, nodo):
        #Recorre el arbol desde un nodo para obtener los indices de los vectores
        if self.tree[nodo]['hijo1'] is None and self.tree[nodo]['hijo2'] is None:
            #print(nodo)
            return [nodo]
        else:
            nodos_finales = []
            if self.tree[nodo]['hijo1'] is not None:
                nodos_finales.extend(self.obtener_nodos_finales(self.tree[nodo]['hijo1']))
            if self.tree[nodo]['hijo2'] is not None:
                nodos_finales.extend(self.obtener_nodos_finales(self.tree[nodo]['hijo2']))
            return nodos_finales
    def predict(self,vector):
        #Pre: Se da un vector
        #Post: Devuelve el label de a que cluster pertenece
        distancia=99999
        nodo=None
        for x in self.centroides.keys():
            nueva_dist = distance.minkowski(vector, self.centroides[x], p=self.grado)
            #nueva_dist = np.linalg.norm(np.array(vector) - np.array(self.centroides[x]))
            if (nueva_dist<distancia):
                distancia=nueva_dist
                nodo=x
                label=x
        print("El punto:"+" pertenece al cluster:"+str(nodo)+" con una distancia de:"+str(distancia))
        return(label)
    def predict_multiple(self,vectors):
        #Pre: Se da una lista de vectores
        #Post: Devuelve un diccionario que asocia cada indice de la lista de vectores con el cluster al que pertenece
        labels={}
        i=0
        for x in vectors:
            labels[i]=self.predict(x)
            i+=1
        return(labels)
    def añadir_linkage(self,linkage,nodo):
        if(self.tree[nodo]['hijo1'] is not None):
            linkage.append([int(self.tree[nodo]['hijo1']),int(self.tree[nodo]['hijo2']),float(self.tree[nodo]['distancia']),int(len(self.obtener_nodos_finales(nodo)))])
            linkage=self.añadir_linkage(linkage,(self.tree[nodo]['hijo1']))
            linkage=self.añadir_linkage(linkage,(self.tree[nodo]['hijo2']))
        return(linkage)
   
    
    def calcular_centroide(self,indice):
        cluster=self.obtener_nodos_finales(indice)
        clusters=[]
        for x in cluster:
            clusters.append(self.vectors[x])
        cluster=clusters
        num_vectores = len(cluster)
        num_caracteristicas = len(cluster[0])  # Suponemos que todos los vectores tienen la misma longitud

        centroide = [0] * num_caracteristicas

        for vector in cluster:
            for i in range(num_caracteristicas):
                centroide[i] += vector[i] / num_vectores

        self.centroides[indice]=centroide
    def buscar_nodos(self,num_clusters=4,dist_max=20):
        #Pre: Se da el número de clusters objetivo o la distancia maxima objetivo
        #Post: Devuelve los nodos en los que se cumplen los requisitos.
        self.dist_max=dist_max
        self.num_clusters=num_clusters
        self.clusters=[self.nodoPadre]
        hemosLlegado=False
        while len(self.clusters)<4 and not hemosLlegado:
            dist=0
            siguiente_nodo=None
            print(self.clusters)
            for x in self.clusters:
                distancia=0
                if self.tree[x]['distancia']>dist and self.tree[x]['distancia']> self.dist_max:
                    siguiente_nodo=max(self.clusters)
            print("El siguiente nodo:"+str(siguiente_nodo))
            if(siguiente_nodo) is None:
                hemosLlegado=True
            else:
                self.clusters.remove(siguiente_nodo)
                self.clusters.append(self.tree[siguiente_nodo]['hijo1'])
                self.clusters.append(self.tree[siguiente_nodo]['hijo2'])
                
            #print(self.clusters)
        
        for x in self.clusters:
            print("El cluster "+str(x)+" contiene estos vectores:")
            vectores=[]
            for y in self.obtener_nodos_finales(x):
                print(self.vectors[y])
                vectores.append(self.vectors[y])
            self.calcular_centroide(x)     
        print("Los centroides son:"+str(self.centroides))
        #self.draw_dendrogram()

    def save(self):
        #Funcion para guardar el modelo
        filename = f"{self.distance_type}_{self.num_clusters}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Guardado en {filename}")

    @classmethod
    def load(cls, filename):
        #Función para cargar el modelo
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        print(f"Cargado desde {filename}")
        return obj
class hierarchical_clustering:
    def __init__(self, vectors, inter_distance_type,p=2):
        #CONSTRUCTORA DE LA CLASE DE ENTRENAMIENTO
        self.iters=0
        self.grado=p
        self.vectors = vectors
        self.tree={}
        self.centroides={}
        self.distance_type = inter_distance_type
        if (self.distance_type=="average"):
            for i in range(len(vectors)):
             self.centroides[i]=self.vectors[i]
        self.clusters = [i for i in range(len(vectors))]
        
        #self.distances = [[self.distance(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))]

    def calcular_centroide(self,indice):
        #Pre:Dado un indice obtiene todos los vectores que contiene
        #Post:Devuelve el centroide asociado a los vectores pertenecientes al cluster
        cluster=self.obtener_nodos_finales(indice)
        clusters=[]
        for x in cluster:
            clusters.append(self.vectors[x])
        cluster=clusters
        num_vectores = len(cluster)
        num_caracteristicas = len(cluster[0])  # Suponemos que todos los vectores tienen la misma longitud

        centroide = [0] * num_caracteristicas

        for vector in cluster:
            for i in range(num_caracteristicas):
                centroide[i] += vector[i] / num_vectores

        self.centroides[indice]=centroide
    def calcular_distancias_NuevoNodo(self,num_nodos):
        #Pre:Se le aporta el nodo recien creado
        #Post:Actualiza la distancia al resto de clusters posibles
        distancia=9999
        for x in self.clusters:
            #print("Se esta recalculando distancia de "+ str(x)+" y " + str(num_nodos))
            if(distance_type!="average"):
                distancia=self.distancia_intracluster(self.clusters_ind[num_nodos],self.clusters_ind[x])
            self.distancias[num_nodos,x]=distancia
            self.distancias[x,num_nodos]=distancia
        return
    def actualizar_arbol(self, nodo1, nodo2, distancia):
        #Pre:Dado los hijos y la distancia actualiza el arbol
        #Post:El arbol se verá incrementado con un nuevo nodo
        num_nodos = len(self.vectors)+self.iters 
        nueva_distancia =  0
        self.clusters_ind[num_nodos] = self.clusters_ind.get(nodo1, []) + self.clusters_ind.get(nodo2, [])
        print(self.clusters_ind[num_nodos])
        self.clusters.remove(nodo1)
        self.clusters.remove(nodo2)
        self.calcular_distancias_NuevoNodo(num_nodos)
        self.clusters.append(num_nodos)
        
        
        # Crear el nuevo nodo
        nuevo_nodo = {
            'hijo1': nodo1,
            'hijo2': nodo2,
            'distancia': distancia,
            'iteracion': self.iters #ESTO PARA NUM CLUSTERS!!
            #'padre':None,
            #'vector': None  # Aquí deberías definir el vector del nuevo nodo si es necesario
        }
        print("nuevo nodo" + str(nuevo_nodo))
        # Añadir el nuevo nodo al árbol
        self.tree[num_nodos] = nuevo_nodo
        if(self.distance_type=="average"):
            self.centroides.pop(nodo1)
            self.centroides.pop(nodo2)
            self.calcular_centroide(num_nodos)

        # Incrementar el contador de iteraciones

        return num_nodos
    def obtener_nodos_finales(self, nodo):
        #Recorre el arbol para obtener los indices de los vectores
        if self.tree[nodo]['hijo1'] is None and self.tree[nodo]['hijo2'] is None:
            #print(nodo)
            return [nodo]
        else:
            nodos_finales = []
            if self.tree[nodo]['hijo1'] is not None:
                nodos_finales.extend(self.obtener_nodos_finales(self.tree[nodo]['hijo1']))
            if self.tree[nodo]['hijo2'] is not None:
                nodos_finales.extend(self.obtener_nodos_finales(self.tree[nodo]['hijo2']))
            return nodos_finales

    def generar_diccionario(self):
        #Genera la base del arbol puesto que esto dará el indice para acceder a los vectores
        resultado = {}
        
        for i, array in enumerate(self.vectors):
            id_unico = i  
            resultado[id_unico] = {'hijo1': None, 'hijo2': None, 'distancia': 0, 'iteracion': -1}
        return resultado
    
    def generar_distancias(self):
        #Pre dado los vectores
        #Post genera las tuplas de distancias entre todos los puntos para evitar recalcularlas
        n = len(self.vectors)
        vectores=[np.array(vec) for vec in self.vectors]
        distancias = {}
        self.clusters_ind={}
        for i in range(n):
            self.clusters_ind[i]=[i]
            for j in range(i+1, n):
                distancia =  distance.minkowski(vectores[i],vectores[j],self.grado)
                #distancia = np.linalg.norm(vectores[i] - vectores[j])
                distancias[(i, j)] = distancia
                distancias[(j, i)] = distancia
        return distancias
    
    ##Aqui se hacen las distancias
    def mean_link(self,cluster1, cluster2):
        total_distancia = 0
        num_pares = 0
        for punto1 in cluster1:
            for punto2 in cluster2:
                total_distancia += self.distancias[(punto1, punto2)]
                num_pares += 1
        return total_distancia / num_pares
    def single_link(self,cluster1, cluster2):
        min_distancia = float('inf')
        for punto1 in cluster1:
            for punto2 in cluster2:
                distancia = self.distancias[(punto1, punto2)]
                if distancia < min_distancia:
                    min_distancia = distancia
        return min_distancia
    def complete_link(self,cluster1, cluster2):
        max_distancia = 0
        for punto1 in cluster1:
            for punto2 in cluster2:
                distancia = self.distancias[(punto1, punto2)]
                if distancia > max_distancia:
                    max_distancia = distancia
        return max_distancia
    def average_link(self,cluster1, cluster2):
        total_distancia = 0
        num_pares = 0
        for punto1 in cluster1:
            for punto2 in cluster2:
                total_distancia += self.distancias[(punto1, punto2)]
                num_pares += 1
        return total_distancia / num_pares
    def mean_link(self):
        centroides = list(self.centroides.keys())
        coordenadas = list(self.centroides.values())
        
        # Calcula todas las distancias entre pares de centroides
        
        distancia_minima = float('inf')
        centroides_mas_cercanos = (None, None)
        
        # Encuentra los centroides más cercanos
        for i in range(len(centroides)):
            for j in range(i + 1, len(centroides)):
                distancia_actual = np.linalg.norm(np.array(coordenadas[i]) - np.array(coordenadas[j]))
                if distancia_actual < distancia_minima:
                    distancia_minima = distancia_actual
                    centroides_mas_cercanos = (centroides[i], centroides[j])
                    centroide1=centroides[i]
                    centroide2=centroides[j]
        
    
        return centroide1,centroide2, distancia_minima
        

    def distancia_intracluster(self,ind_vect_cluster1,ind_vect_cluster2):
        #Redirige para saber que tipo de disttancia intracluster usar
        #'single','complete','average','mean'
        if(self.distance_type=="single"):
            distancia=self.single_link(ind_vect_cluster1,ind_vect_cluster2)
        elif(self.distance_type=="mean"):
            distancia=self.average_link(ind_vect_cluster1,ind_vect_cluster2)
        elif(self.distance_type=="average"):
            #distancia=self.mean_link(ind_vect_cluster1,ind_vect_cluster2)

            print("Esto no se deberia ejecutar nunca puesto que se trata en otro momento")
        else:
            distancia=self.complete_link(ind_vect_cluster1,ind_vect_cluster2)
        #print("Distancia entre cluster:"+str(ind_vect_cluster1)+" y cluster:"+str(ind_vect_cluster2)+" :"+str(distancia))
        
        return(distancia)
    
    def encontrar_clusters_mas_cercanos(self):
        #Pre: Tenemos una lista de cluster mayor que 1
        #Post: Devuelve los dos nodos con distancia intergrupal menor para posteriormente unirlos llamando a la función añadir_arbol
        if(self.distance_type=="average"):
           indice_cluster1, indice_cluster2,distancia_minima=self.mean_link()
        else:
            distancia_minima = float('inf')
            indice_cluster1 = None
            indice_cluster2 = None
                
            for i in range(len(self.clusters)):
                cluster1 = self.clusters[i]
                vectores_cluster1=self.obtener_nodos_finales(cluster1)
                for j in range(i + 1, len(self.clusters)):
                    
                    cluster2 = self.clusters[j]
                    
                    if ((self.clusters[i],self.clusters[j]) not in self.distancias):
                        
                        vectores_cluster2=self.obtener_nodos_finales(cluster2)
                        
                        distancia = self.distancia_intracluster(vectores_cluster1, vectores_cluster2)
                    else:
                        #print("Distancia precalculada")
                        distancia=self.distancias[(self.clusters[i],self.clusters[j])]
                    
                    if distancia < distancia_minima:
                        distancia_minima = distancia
                        indice_cluster1 = cluster1
                        indice_cluster2 = cluster2
                    elif distancia == distancia_minima:
                        if random.choice([True, False]):
                            indice_cluster1 = cluster1
                            indice_cluster2 = cluster2
            print("Los nodos más cercanos son:"+str(indice_cluster1)+" y "+str(indice_cluster2))
            
        return indice_cluster1, indice_cluster2,distancia_minima

         
    def cluster(self):
        #Funcion que inicia el entrenamiento, dado una lista de vectores esta se ira agrupando hasta quedar solo 1 un cluster, se verá reflejado en el arbol y podra recorrerse con una clase auxiliar
        if self.iters == 0:
            self.tree = self.generar_diccionario()
            self.distancias=self.generar_distancias()
              
        while len(self.clusters) > 1:
            #nodo1, nodo2 = self.encontrar_nodos_mas_cercanos()
            #self.actualizar_arbol(nodo1, nodo2)
            nodo1,nodo2,distancia=self.encontrar_clusters_mas_cercanos()
            self.actualizar_arbol(nodo1,nodo2,distancia)
            self.iters += 1
        proc=procesarCluster(self.vectors,self.tree,distance_type=self.distance_type,p=2)
        
        return proc
    def añadir_linkage(self,linkage,nodo):
        #Metodo para hacer la matriz linkage que representará el dendrograma.
        if distance_type=="average":
            if(self.tree[nodo]['hijo1'] is not None):
                linkage.append([int(self.tree[nodo]['hijo1']),int(self.tree[nodo]['hijo2']),float(self.tree[nodo]['distancia']+float(self.tree[nodo]["iteracion"])),int(len(self.obtener_nodos_finales(nodo)))])
                self.añadir_linkage(linkage,(self.tree[nodo]['hijo1']))
                self.añadir_linkage(linkage,(self.tree[nodo]['hijo2']))
            return(linkage)
        else:
            if(self.tree[nodo]['hijo1'] is not None):
                linkage.append([int(self.tree[nodo]['hijo1']),int(self.tree[nodo]['hijo2']),float(self.tree[nodo]['distancia']),int(len(self.obtener_nodos_finales(nodo)))])
                self.añadir_linkage(linkage,(self.tree[nodo]['hijo1']))
                self.añadir_linkage(linkage,(self.tree[nodo]['hijo2']))
            return(linkage)
    def draw_dendrogram(self):
        # Obtener las distancias y las uniones
        linkage=[]
        nodos_base=len(self.vectors)
        """"
        for x in self.clusters:
            linkage=self.añadir_linkage(linkage,x)
        print("---------------------------------")
        print(self.clusters)
        print(linkage)
        linkage= sorted(linkage, key=lambda x: x[2])
        """
        linkage=[]
        for clave in sorted(self.tree.keys(), reverse=True):
            #print(clave)
            if(self.tree[clave]['hijo1']is not None):
                linkage.append([int(self.tree[clave]['hijo1']),int(self.tree[clave]['hijo2']),float(self.tree[clave]['distancia']),int(len(self.obtener_nodos_finales(clave)))])
       
        print("------------------------")
        linkage=linkage[::-1]
        print(linkage)
       
    
        # Crear el dendrograma
        plt.figure(figsize=(10, 5))
        if self.distance_type=="average":
            for x in range(len(linkage)):
                linkage[x][2]=1.00
            dendrogram(linkage,labels=list(range(len(linkage)+1)),distance_sort=False, leaf_rotation=90, leaf_font_size=8, orientation='top')
        else:
            dendrogram(linkage,labels=list(range(len(linkage)+1)), leaf_rotation=90, leaf_font_size=8, orientation='top')
        plt.xlabel('Índices de los clusters')
        plt.ylabel('Distancia')
        plt.title('Dendrograma '+'Distancia:'+self.distance_type)
        plt.show()

    def draw_tree(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Dibuja los nodos
        for node_id, node_info in self.tree.items():
            if node_info['hijo1'] is not None:
                x1, y1 = node_id, node_info['iteracion']
                x2, y2 = node_info['hijo1'], self.tree[node_info['hijo1']]['iteracion']
                ax.plot([x1, x2], [y1, y2], 'k-', lw=2)
            if node_info['hijo2'] is not None:
                x1, y1 = node_id, node_info['iteracion']
                x2, y2 = node_info['hijo2'], self.tree[node_info['hijo2']]['iteracion']
                ax.plot([x1, x2], [y1, y2], 'k-', lw=2)
            
            ax.plot(node_id, node_info['iteracion'], 'ko', ms=10)
            
            # Agrega la distancia en azul y en un tamaño más grande
            if node_info['distancia'] > 0:
                ax.text(node_id, node_info['iteracion'], f'{node_info["distancia"]:.2f}', ha='center', va='center', color='blue', fontsize=12)
        
        plt.show()


    def export(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
if __name__ == "__main__":
    vectors = [[142, 120, 47, 4, 37],
 [34.3434, 187, 68, 145, 5],
 [138, 83, 185, 26, 176],
 [66, 134, 96, 168, 149],
 [64, 58, 199.34343, 77, 175],
 [101, 138, 170, 16, 62],
 [189, 128, 189, 13, 99],
 [41, 14, 109, 184, 17],
 [32, 169.433434, 41, 69, 91]]
    help()
    # Cargamos el modelo
    model = Doc2Vec.load('my_doc2vec.model')
    # model = gensim.downloader.load('glove-twitter-25')
    
    # Cargamos los datos
    train_df = pd.read_csv('verbalAutopsy_train.csv')
   
    
    # Obtenemos la vectorizaciÃ³n de los documentos
    train_corpus = list(vec_docEmbeddings(train_df["open_response"], model))
    vectors = train_corpus[400:700]
    test=train_corpus[900:1000]
 
    for distance_type in ['complete','single','mean','average']:
        hc = hierarchical_clustering(vectors, distance_type,p=3)
        proc = hc.cluster()
        proc.buscar_nodos(num_clusters=4,dist_max=0)
        labels=proc.predict_multiple(test)
        print(labels)
        proc.save()
        #print(hc.clusters_ind[500])
        #print(hc.obtener_nodos_finales(500))
        #print(merge_history)
        #print(hc.vectors)
        #hc.draw_tree()
        
        hc.draw_dendrogram()
        #hc.obtener_nodos_finales(11)
        #hc.print_clusters()

   