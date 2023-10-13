from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
class hierarchical_clustering:
    def __init__(self, vectors, inter_distance_type):
        self.vectors = vectors
        self.clusters = [{i} for i in range(len(vectors))]
        self.distance_type = inter_distance_type
        self.distances = [[self.distance(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))]

    def distance(self, vec1, vec2):
        if self.distance_type == 'single':
            return self.single_linkage_distance(vec1, vec2)
        elif self.distance_type == 'complete':
            return self.complete_linkage_distance(vec1, vec2)
        elif self.distance_type == 'average':
            return self.average_linkage_distance(vec1, vec2)

    def single_linkage_distance(self, vec1, vec2):
        print("Usando distancia single-link")
        return min(self.distance_single(vec1[i], vec2[i]) for i in range(len(vec1)))

    def complete_linkage_distance(self, vec1, vec2):
        print("Usando distancia complete-link")
        return max(self.distance_single(vec1[i], vec2[i]) for i in range(len(vec1)))

    def average_linkage_distance(self, vec1, vec2):
        print("Usando distancia avg-link")
        return sum(self.distance_single(vec1[i], vec2[i]) for i in range(len(vec1))) / len(vec1)

    def distance_single(self, x, y):
        return abs(x - y)

    def merge_clusters(self, cluster1, cluster2):
        new_cluster = cluster1.union(cluster2)
        return new_cluster

    def find_closest_clusters(self):
        min_distance = float('inf')
        min_i, min_j = None, None

        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                # Asegurémonos de que los índices estén en los límites de la matriz de distancias
                if i < len(self.distances) and j < len(self.distances[i]):
                    if self.distances[i][j] < min_distance:
                        min_distance = self.distances[i][j]
                        min_i, min_j = i, j

        return min_i, min_j
    def cluster(self):
        merge_history = []  # Lista para almacenar historial de fusiones (índices y distancias)
        
        while len(self.clusters) > 1:
            i, j = self.find_closest_clusters()
            new_cluster = self.merge_clusters(self.clusters[i], self.clusters[j])
            
            # Guardar información sobre la fusión
            merge_history.append((i, j, self.distances[i][j]))
            
            self.clusters.pop(j)
            self.distances.pop(j)
            for row in self.distances:
                row.pop(j)
            self.clusters[i] = new_cluster

        # Obtener los vectores correspondientes a los índices en el cluster final
        cluster_indices = list(self.clusters[0])
        result = [self.vectors[i] for i in cluster_indices]
        return result, merge_history



    def plot_dendrogram(self,uniones):
        plt.figure(figsize=(10, 5))

        # Inicializar un diccionario para rastrear las posiciones de los clusters
        cluster_pos = {}

        # Ordenar las uniones por distancia
        merge_history.sort(key=lambda x: x[2])

        for i, (cluster1, cluster2, distancia) in enumerate(merge_history):
            # Obtener las posiciones de los clusters fusionados
            x1 = cluster_pos.get(cluster1, cluster1)
            x2 = cluster_pos.get(cluster2, cluster2)
            
            # Calcular la posición del punto medio
            x_mid = (x1 + x2) / 2

            # Guardar la posición del nuevo cluster fusionado
            cluster_pos[i + len(merge_history)] = x_mid

            # Dibujar una línea vertical que conecta los clusters
            plt.plot([x1, x1, x2, x2], [0, distancia, distancia, 0], 'k-', lw=0.5)

            # Dibujar una línea horizontal que representa la distancia
            plt.plot([x1, x2], [distancia, distancia], 'k-', lw=0.5)

            # Dibujar un punto en el punto medio
            plt.plot(x_mid, distancia, 'ko', markersize=3)

            # Etiquetar la distancia
            plt.text(x_mid, distancia, f'{distancia:.2f}', va='bottom', ha='center', fontsize=8)

        # Establecer límites y etiquetas
        plt.xlim(-1, len(merge_history) * 2 + 1)
        plt.xlabel('Clusters')
        plt.ylabel('Distancia')
        plt.title('Dendrograma Jerarquico usando '+ self.distance_type + ' distance')

        plt.show()



if __name__ == "__main__":
    vectors = [[142, 120, 47, 4, 37],
 [34.3434, 187, 68, 145, 5],
 [138, 83, 185, 26, 176],
 [66, 134, 96, 168, 149],
 [64, 58, 199.34343, 77, 175],
 [101, 138, 170, 16, 62],
 [189, 128, 189, 13, 99],
 [41, 14, 109, 184, 17],
 [32, 169.433434, 41, 69, 91],]

 
    for distance_type in ['single','complete','average']:
        hc = hierarchical_clustering(vectors, distance_type)
        
        result, merge_history = hc.cluster()
        #print(result)
        print(merge_history)
        hc.plot_dendrogram(merge_history)

   
