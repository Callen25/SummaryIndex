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
ir_model = "DSF"


def get_embedding(text):
    return sentence_model([text])[0]


# For each distance in a list of euclidean distances,
# Generates a score from euclidean distance using the following formula:
# Score = 1 / (1 + distance)
def distance_to_score(distances):
    scores = []
    for distance in distances:
        score = 1 / (1 + distance)
        scores.append(score)
    return scores


# Convert list of ids to doc_ids
def id_to_docid(ids):
    doc_ids = []
    for id in ids:
        doc_ids.append(id_map[id][0])
    return doc_ids


def group_by_docid(doc_ids, scores):
    doc_id_map = dict()
    # Assign each entry to an empty list
    for doc_id in doc_ids:
        doc_id_map[doc_id] = list()
    # For each doc_id add its associated scores
    for i in range(len(scores)):
        doc_id_map[doc_ids[i]].append(scores[i])
    # For each entry sort its scores in descending order
    for doc_id in doc_id_map:
        doc_id_map[doc_id].sort(reverse=True)
    return doc_id_map


def aggregate_scores(doc_id_scores):
    for doc_id in doc_id_scores:
        doc_id_scores[doc_id] = get_aggregate_score(doc_id_scores[doc_id])
    return list(doc_id_scores.keys()), list(doc_id_scores.values())


def get_aggregate_score(score_list):
    aggregate_score = 0
    for score in score_list:
        aggregate_score = (1 - aggregate_score) * score
    return aggregate_score


def parse_results(result):
    # Get scores from euclidean distance
    scores = distance_to_score(result[1])
    # Map id to doc_id
    doc_ids = id_to_docid(result[2])
    # Group scores by doc id in descending order
    doc_id_scores = group_by_docid(doc_ids, scores)
    # Get aggregated score
    doc_ids, scores = aggregate_scores(doc_id_scores)
    return doc_ids, scores


def get_closest(query_vector, range):
    results = index.range_search(query_vector, range)
    return parse_results(results)


def write_query_result(results_file, query_id, scores, doc_ids):
    for i in range(len(doc_ids)):
        results_file.write(f"{query_id} Q0 {doc_ids[i]} {i + 1} {scores[i]} {ir_model}\n")


def main():
    global index, id_map, query_file
    queries = query_file.readlines()
    for threshold in thresholds:
        index = faiss.read_index(f"index/faiss{threshold}.index")
        id_map = pickle.load(open(f"index/map{threshold}.p", "rb"))

        for range in distance_ranges:
            results_file = open(f"eval/results/range_{range}_thresh_{threshold}.txt", "w")
            for line in queries:
                split_line = line.split(":::")
                query_id = int(split_line[0])
                query = split_line[1]
                query_vector = np.array([get_embedding(query)])
                doc_ids, scores = get_closest(query_vector, range)
                write_query_result(results_file, query_id, scores, doc_ids)
            results_file.close()


if __name__ == '__main__':
    main()
