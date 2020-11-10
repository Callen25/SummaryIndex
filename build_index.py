import os
import pickle
import faiss
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

VECTOR_DIR = "vectors"
k = 3
VECTOR_DIMENSION = 768
index = faiss.IndexFlatIP(VECTOR_DIMENSION)
SENTENCE_TRANSFORMER = SentenceTransformer('bert-base-nli-mean-tokens')
current_id = 0
id_map = dict()


def build_index(file_name):
    global current_id, index
    vectors = pickle.load(open(file_name, "rb"))
    doc_id = file_name.split(".")[0]
    for vector in vectors:
        index.add(np.array([vector]))
        id_map[current_id] = doc_id
        current_id += 1


def get_embedding(text):
    tokenized_query = list()
    tokenized_query.append(word_tokenize(text))
    return SENTENCE_TRANSFORMER.encode(tokenized_query)[0]


for i, file in enumerate(os.listdir(VECTOR_DIR)):
    file_path = os.path.join(VECTOR_DIR, file)
    build_index(file_path)

while True:
    query = input("Find Closest Document: ")
    query_vector = get_embedding(query)
    result = (index.search(np.array([query_vector]), k))
    for val in result[1]:
        for test in val:
            print("{} maps to {}".format(test, id_map[test]))
