def train_model()
    return 


def main():
    # Distance type: ['complete','single','mean','average']
    hc = hierarchical_clustering(vectors, 'complete', p=3)
    proc = hc.cluster() # fit()
    
    proc.cortar_arbol(num_clusters=3,dist_max=0)
    proc.draw_dendrogram() # Dibujar arbol completo
    
    labels = proc.predict_multiple(vectors) # Obtener labels por cada vector
        
    proc.save() # Guardar el modelo
    
    
    proc2=hierarchical_clustering.load("complete_3.pkl")
    proc2.cortar_arbol(num_clusters=4,dist_max=0)
    proc2.draw_dendrogram()
    print(labels)


if __name__ == "__main__":
    

    main()
        exit(0)
