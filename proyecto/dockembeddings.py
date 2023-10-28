from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec
import gensim.downloader
import smart_open

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd

import os

def preprocesado(texto):
    # PRE: Un texto
    # POST: Tockens del texto preprocesado
    return simple_preprocess(texto)

def train_docModel(texts, model_file):
    # Inicializamos y entrenamos un modelo Doc2Vec
    #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    train_df = pd.read_csv(train_file)
    documents = [TaggedDocument(preprocesado(doc), [i]) for i, doc in enumerate(texts)]

    model = Doc2Vec(documents, vector_size=200, window=2, min_count=1, workers=4)

    # Guardamos el modelo
    model.save(model_file)

def load_docModel(model_file):
    # Cargamos el modelo
    # model = gensim.downloader.load('glove-twitter-25')
    model = Doc2Vec.load(model_file)
    return model

def vec_docEmbeddings(docs, model):
    # PRE: una lista de documentos y el modelo de doc-embeddings
    # POST: lista de (id, vector)
    for i, line in enumerate(docs):
        # Preprocesado del documento
        tokens = preprocesado(line)
        # Vectorizamos -> indice, vector
        yield (i, model.infer_vector(tokens))