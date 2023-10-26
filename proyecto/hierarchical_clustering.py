from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import random
#import networkx as nx
import pickle




# BORRAR EN FUTURO
"""
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec
import gensim.downloader
import smart_open

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd

import os
from dockembeddings import vec_docEmbeddings
"""
class procesarCluster():
    def __init__(self,vectors,arbol,num_clusters=4,dist_max=20,distance_type='single'):
        self.distance_type=distance_type
        self.vectors=vectors
        self.tree=arbol
        self.num_clusters=num_clusters
        self.dist_max=dist_max
        self.nodoPadre=max(self.tree.keys())
        self.clusters=[]
        self.centroides={}
    def obtener_nodos_finales(self, nodo):
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
        distancia=99999
        nodo=None
        for x in self.centroides.keys():
            nueva_dist = np.linalg.norm(np.array(vector) - np.array(self.centroides[x]))
            if (nueva_dist<distancia):
                distancia=nueva_dist
                nodo=x
        print("El punto:"+str(vector)+" pertenece al cluster:"+str(nodo)+" con una distancia de:"+str(distancia))
    def añadir_linkage(self,linkage,nodo):
        if(self.tree[nodo]['hijo1'] is not None):
            linkage.append([int(self.tree[nodo]['hijo1']),int(self.tree[nodo]['hijo2']),float(self.tree[nodo]['distancia']),int(len(self.obtener_nodos_finales(nodo)))])
            linkage=self.añadir_linkage(linkage,(self.tree[nodo]['hijo1']))
            linkage=self.añadir_linkage(linkage,(self.tree[nodo]['hijo2']))
        return(linkage)
    def draw_dendrogram(self):
        return
        # Obtener las distancias y las uniones
        linkage=[]
        for x in self.clusters:
            linkage=self.añadir_linkage(linkage,x)
        print("---------------------------------")
        print(self.clusters)
            
        linkage= sorted(linkage, key=lambda x: x[2])
        print(linkage)
        # Crear el dendrograma
        plt.figure(figsize=(10, 5))
        dendrogram(linkage,labels=list(range(len(linkage)+1)), leaf_rotation=90, leaf_font_size=8, orientation='top')
        plt.xlabel('Índices de los clusters')
        plt.ylabel('Distancia')
        plt.title('Dendrograma '+'Distancia:'+self.distance_type)
        plt.show()

    
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
                
            print(self.clusters)
        
        for x in self.clusters:
            print("El cluster "+str(x)+" contiene estos vectores:")
            vectores=[]
            for y in self.obtener_nodos_finales(x):
                print(self.vectors[y])
                vectores.append(self.vectors[y])
            self.calcular_centroide(x)     
        print("Los centroides son:"+str(self.centroides))
        self.draw_dendrogram()

class hierarchical_clustering:
    def __init__(self, vectors, inter_distance_type):
        self.iters=0
        self.vectors = vectors
        self.tree={}
        self.clusters = [i for i in range(len(vectors))]
        self.distance_type = inter_distance_type
        #self.distances = [[self.distance(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))]
    
    def actualizar_arbol(self, nodo1, nodo2, distancia):
        num_nodos = len(self.vectors)+self.iters 
        nueva_distancia =  0
        self.clusters.remove(nodo1)
        self.clusters.remove(nodo2)
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

        # Incrementar el contador de iteraciones

        return num_nodos
    def obtener_nodos_finales(self, nodo):
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
        resultado = {}
        for i, array in enumerate(self.vectors):
            id_unico = i  # Incrementamos en 1 para que los IDs comiencen en 1
            resultado[id_unico] = {'hijo1': None, 'hijo2': None, 'distancia': 0, 'iteracion': -1}
        return resultado
    
    def generar_distancias(self):
        n = len(self.vectors)
        vectores=[np.array(vec) for vec in self.vectors]
        distancias = {}

        for i in range(n):
            for j in range(i+1, n):
                distancia = np.linalg.norm(vectores[i] - vectores[j])
                distancias[(i, j)] = distancia
                distancias[(j, i)] = distancia
        return distancias
    
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

    def distancia_intracluster(self,ind_vect_cluster1,ind_vect_cluster2):
        #'single','complete','average','mean'
        if(self.distance_type=="single"):
            distancia=self.single_link(ind_vect_cluster1,ind_vect_cluster2)
        elif(self.distance_type=="average"):
            distancia=self.average_link(ind_vect_cluster1,ind_vect_cluster2)
        elif(self.distance_type=="mean"):
            distancia=self.mean_link(ind_vect_cluster1,ind_vect_cluster2)
        else:
            distancia=self.complete_link(ind_vect_cluster1,ind_vect_cluster2)
        #print("Distancia entre cluster:"+str(ind_vect_cluster1)+" y cluster:"+str(ind_vect_cluster2)+" :"+str(distancia))
        
        return(distancia)
    
    def encontrar_clusters_mas_cercanos(self):
        distancia_minima = float('inf')
        indice_cluster1 = None
        indice_cluster2 = None
            
        for i in range(len(self.clusters)):
            cluster1 = self.clusters[i]
            vectores_cluster1=self.obtener_nodos_finales(cluster1)
            for j in range(i + 1, len(self.clusters)):
                
                cluster2 = self.clusters[j]
                
                vectores_cluster2=self.obtener_nodos_finales(cluster2)
                
                distancia = self.distancia_intracluster(vectores_cluster1, vectores_cluster2)
                
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
        if self.iters == 0:
            self.tree = self.generar_diccionario()
            self.distancias=self.generar_distancias()
              
        while len(self.clusters) > 1:
            #nodo1, nodo2 = self.encontrar_nodos_mas_cercanos()
            #self.actualizar_arbol(nodo1, nodo2)
            nodo1,nodo2,distancia=self.encontrar_clusters_mas_cercanos()
            self.actualizar_arbol(nodo1,nodo2,distancia)
            self.iters += 1
        proc=procesarCluster(self.vectors,self.tree,distance_type=self.distance_type)
        
        return self.tree,proc
    
    def añadir_linkage(self,linkage,nodo):
        if(self.tree[nodo]['hijo1'] is not None):
            linkage.append([int(self.tree[nodo]['hijo1']),int(self.tree[nodo]['hijo2']),float(self.tree[nodo]['distancia']),int(len(self.obtener_nodos_finales(nodo)))])
            self.añadir_linkage(linkage,(self.tree[nodo]['hijo1']))
            self.añadir_linkage(linkage,(self.tree[nodo]['hijo2']))
        return(linkage)
    
    def draw_dendrogram(self):
        # Obtener las distancias y las uniones
        linkage=[]
        nodos_base=len(self.vectors)
        for x in self.clusters:
            linkage=self.añadir_linkage(linkage,x)
        print("---------------------------------")
        print(self.clusters)
        print(linkage)    
        linkage= sorted(linkage, key=lambda x: x[2])
    
        # Crear el dendrograma
        plt.figure(figsize=(10, 5))
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
    """
    # Cargamos el modelo
    model = Doc2Vec.load('my_doc2vec.model')
    # model = gensim.downloader.load('glove-twitter-25')
    
    # Cargamos los datos
    train_df = pd.read_csv('verbalAutopsy_train.csv')
    
    
    # Obtenemos la vectorizaciÃ³n de los documentos
    train_corpus = list(vec_docEmbeddings(train_df["open_response"], model))
    vectors = train_corpus[0:200]
    """
    for distance_type in ['single','complete','average','mean']:
        hc = hierarchical_clustering(vectors, distance_type)
        merge_history,proc = hc.cluster()
        proc.buscar_nodos(num_clusters=10,dist_max=140)
        #proc.predict([32, 169.433434, 41, 69, 91])
        print(merge_history)
        print(hc.vectors)
        #hc.draw_tree()
        hc.draw_dendrogram()
        #hc.obtener_nodos_finales(11)
        #hc.print_clusters()

   