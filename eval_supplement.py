import faiss
import pickle
import numpy as np
import tensorflow_hub as hub
import json
import urllib.parse
import urllib.request

MAX_DIST = 1.5674068

index = None
id_map = None
THRESHOLD = 0.95
RANGE = 1.205
sentence_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
query_file = open("eval/queries.txt", "r")
ir_model = "DSF"


def get_embedding(text):
    return sentence_model([text])[0]


# For each distance in a list of euclidean distances,
# Generates a normalized score from the euclidean distance
def distance_to_score(distances):
    scores = []
    for distance in distances:
        score = 1 - (distance / MAX_DIST)
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
    doc_ids = []
    scores = []
    for doc_id in doc_id_scores:
        doc_id_scores[doc_id] = get_aggregate_score(doc_id_scores[doc_id])

    results = [(k, v) for k, v in doc_id_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)

    if results is not None:
        doc_ids = [i[0] for i in results]
        scores = [i[1] for i in results]
    return doc_ids, scores


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
    return {doc_ids[i]: scores[i] for i in range(len(doc_ids))}


def get_closest(query_vector, range):
    results = index.range_search(query_vector, range)
    return parse_results(results)


def write_query_result(results_file, query_id, scores, doc_ids):
    for i in range(len(doc_ids)):
        results_file.write(f"{query_id} Q0 {doc_ids[i]} {i + 1} {scores[i]} {ir_model}\n")


def solr_search(query):
    query = urllib.parse.quote(query)
    url = 'http://localhost:8983/solr/trec/select?fl=docno%2C%20score&q=doctext%3A(' + query + ')&rows=10&sort=score%20desc'
    data = urllib.request.urlopen(url)
    results = json.load(data)['response']['docs']
    results_dict = {}
    for entry in results:
        results_dict[entry['docno']] = entry['score']
    return results_dict


def weigh_results(solr_results, vector_results, vector_weight):
    # Apply weighting to vector score, and combine with solr results
    for doc_id in vector_results:
        vector_results[doc_id] = vector_results[doc_id] * vector_weight
        if doc_id not in solr_results:
            solr_results[doc_id] = vector_results[doc_id]
        else:
            solr_results[doc_id] = solr_results[doc_id] + vector_results[doc_id]

    # Sort doc_ids by score in descending order
    tuple_list = [(doc_id, score) for doc_id, score in solr_results.items()]
    tuple_list.sort(key=lambda x: x[1], reverse=True)
    # Convert tuple list into list of doc_ids and list of scores
    doc_ids = []
    scores = []
    for doc_id, score in tuple_list:
        doc_ids.append(doc_id)
        scores.append(score)
    return doc_ids, scores


def main():
    global index, id_map, query_file
    queries = query_file.readlines()

    index = faiss.read_index(f"index/faiss{THRESHOLD}.index")
    id_map = pickle.load(open(f"index/map{THRESHOLD}.p", "rb"))

    for weight in range(1, 21):
        results_file = open(f"eval/supplement_results/weight_{weight}.txt", "w")
        for line in queries:
            split_line = line.split(":::")
            query_id = int(split_line[0])
            query = split_line[1]
            query_vector = np.array([get_embedding(query)])
            vector_results = get_closest(query_vector, RANGE)
            solr_results = solr_search(query)
            doc_ids, scores = weigh_results(solr_results, vector_results, weight)
            write_query_result(results_file, query_id, scores, doc_ids)
        results_file.close()


if __name__ == '__main__':
    main()
