import faiss
import pickle
import numpy as np
import tensorflow_hub as hub
import os

index = None
id_map = None
sentence_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
k = 10


def get_embedding(text):
    return sentence_model([text])[0]


def get_doc_ids(result):
    doc_ids = []
    matches = result[1][0]
    for match in matches:
        doc_ids.append(id_map[match][0])
    return doc_ids


def get_closest_match(query_vector):
    val = index.search(query_vector, 1)
    return get_doc_ids(val)[0], val[1][0][0]


def get_k_closest(query_vector):
    global k
    results = index.search(query_vector, k)
    return get_doc_ids(results)


def main():
    global index, id_map
    index = faiss.read_index("index/faiss.index")
    id_map = pickle.load(open("index/map.p", "rb"))
    while True:
        query = input("Query: ").lower().strip()
        print(query)
        query_vector = np.array([get_embedding(query)])
        print("Closest Value: ", get_k_closest(query_vector))


if __name__ == '__main__':
    main()
