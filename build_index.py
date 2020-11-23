import os
import pickle
import faiss
import cupy as cp
import numpy as np

VECTOR_DIR = "vectors_sim0.0"
VECTOR_DIMENSION = 512
thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
index = faiss.IndexFlatL2(VECTOR_DIMENSION)
id_map = list()


def add_vectors(file_name, f):
    global index
    vectors = pickle.load(open(file_name, "rb"))
    doc_id = f.split(".")
    for vector in vectors:
        numpy_vector = cp.asnumpy(cp.asarray([vector]))
        index.add(np.ascontiguousarray(numpy_vector))
        id_map.append(doc_id)


def save_vectors(thr):
    global index, id_map
    faiss.write_index(index, f"index/faiss{thr}.index")
    pickle.dump(id_map, open(f"index/map{thr}.p", "wb"))


for threshold in thresholds:
    VECTOR_DIR = f"vectors_sim{threshold}"
    for file in os.listdir(VECTOR_DIR):
        file_path = os.path.join(VECTOR_DIR, file)
        add_vectors(file_path, file)

    save_vectors(threshold)

    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    id_map = list()
