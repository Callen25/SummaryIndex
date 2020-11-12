import os
import pickle
import faiss
import numpy as np

VECTOR_DIR = "vectors"
VECTOR_DIMENSION = 512
index = faiss.IndexFlatL2(VECTOR_DIMENSION)
id_map = list()


def add_vectors(file_name, f):
    global index
    vectors = pickle.load(open(file_name, "rb"))
    doc_id = f.split(".")
    for vector in vectors:
        numpy_vector = np.array([vector.numpy()])
        index.add(numpy_vector)
        id_map.append(doc_id)


def save_vectors():
    global index, id_map
    faiss.write_index(index, "index/faiss.index")
    pickle.dump(id_map, open("index/map.p", "wb"))


for file in os.listdir(VECTOR_DIR):
    file_path = os.path.join(VECTOR_DIR, file)
    add_vectors(file_path, file)

save_vectors()
