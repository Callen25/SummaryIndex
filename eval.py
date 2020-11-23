import faiss
import pickle
import numpy as np
import tensorflow_hub as hub

index = None
id_map = None
thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
distance_ranges = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
sentence_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
query_file = open("eval/queries.txt", "r")


def get_embedding(text):
    return sentence_model([text])[0]


def get_doc_ids(result):
    doc_ids = []
    matches = result[2]
    for match in matches:
        doc_ids.append(id_map[match][0])
    return doc_ids


def get_closest_match(query_vector):
    val = index.search(query_vector, 1)
    return get_doc_ids(val)[0], val[1][0][0]


def get_closest(query_vector, range):
    results = index.range_search(query_vector, range)
    return get_doc_ids(results)


def main():
    global index, id_map
    for threshold in thresholds:
        index = faiss.read_index(f"index/faiss{threshold}.index")
        id_map = pickle.load(open(f"index/map{threshold}.p", "rb"))
        for range in distance_ranges:
            for line in query_file.readlines():
                split_line = line.split(":::")
                query_id = int(split_line[0])
                query = split_line[1]
                query_vector = np.array([get_embedding(query)])
                results = get_closest(query_vector, range)


if __name__ == '__main__':
    main()
