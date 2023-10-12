from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
class hierarchical_clustering:
    def __init__(self, vectors):
        self.vectors = vectors
        self.clusters = [{i} for i in range(len(vectors))]
        self.distances = [[self.distance(vectors[i], vectors[j]) for j in range(len(vectors))] for i in range(len(vectors))]

    def distance(self, vec1, vec2):
        # Define tu función de distancia aquí (por ejemplo, distancia euclidiana)
        return sum((x - y)**2 for x, y in zip(vec1, vec2))**0.5

    def merge_clusters(self, cluster1, cluster2):
        new_cluster = cluster1.union(cluster2)
        return new_cluster

    def find_closest_clusters(self):
        min_distance = float('inf')
        min_i, min_j = None, None

        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                if self.distances[i][j] < min_distance:
                    min_distance = self.distances[i][j]
                    min_i, min_j = i, j

        return min_i, min_j

    def cluster(self):
        while len(self.clusters) > 1:
            i, j = self.find_closest_clusters()
            new_cluster = self.merge_clusters(self.clusters[i], self.clusters[j])
            self.clusters.pop(j)
            self.distances.pop(j)
            for row in self.distances:
                row.pop(j)
            self.clusters[i] = new_cluster

        # Obtener los vectores correspondientes a los índices en el cluster final
        cluster_indices = list(self.clusters[0])
        result = [self.vectors[i] for i in cluster_indices]
        return result
    def plot_dendrogram(self):
            Z = linkage(self.vectors)
            plt.figure(figsize=(10, 5))
            dendrogram(Z, labels=[str(i) for i in range(len(self.vectors))])
            plt.title("Dendrograma de Clustering Jerárquico")
            plt.xlabel("Índice de la Muestra")
            plt.ylabel("Distancia")
            plt.show()

            hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
            labels = hierarchical_cluster(self.vectors)
            x=[1, 2,  8,  8, 10,  3]
            y=[2, 3, 7, 9, 11, 4]
            plt.scatter(x, y, c=labels)
            plt.show()
if __name__ == "__main__":
    vectors = [[1, 2], [2, 3], [8, 7], [8, 9], [10, 11], [3, 4]]
    hc = hierarchical_clustering(vectors)
    hc.cluster()
    hc.plot_dendrogram()
   