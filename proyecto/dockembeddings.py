from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec
import gensim.downloader
import smart_open

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd

import os

data_dir = "."
model_file = 'my_doc2vec.model'
train_file = 'verbalAutopsy_train.csv'
test_file = 'verbalAutopsy_test.csv'

def preprocesado(texto)
    # PRE: Un texto
    # POST: Tockens del texto preprocesado
    return simple_preprocess(texto)

def train_docModel(texts, model_file):
    # Inicializamos y entrenamos un modelo Doc2Vec
    #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    train_df = pd.read_csv(train_file)
    documents = [TaggedDocument(preprocesado(doc), [i]) for i, doc in enumerate(texts)]

    model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

    # Guardamos el modelo
    model.save(model_file)


def vec_docEmbeddings(docs, model):
    for i, line in enumerate(docs):
        # Preprocesado del documento
        tokens = preprocesado(line)
        # Vectorizamos
        yield model.infer_vector(tokens)


# Entrenamos el modelo
train_docModel(pd.read_csv(train_file)['open_response'], model_file)

# Cargamos el modelo
model = Doc2Vec.load(model_file)
# model = gensim.downloader.load('glove-twitter-25')

# Cargamos los datos
train_df = pd.read_csv(train_file)


# Obtenemos la vectorizaciÃ³n de los documentos
train_corpus = list(vec_docEmbeddings(train_df["open_response"], model))