import os
import xml.etree.ElementTree as ElementTree
from nltk.tokenize import sent_tokenize
import tensorflow_hub as hub
import cupy as np
from cupy.linalg import norm
import pickle

SIM_THRESHOLD = 0.0
MAX_DIST = 1.5674068
thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

DATA_DIR = "data"
sentence_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def euclidean_similarity(a, b):
    distance = norm(a - b)
    return 1 - (distance / MAX_DIST)


def index_document(document):
    doc_root = ElementTree.parse(document).getroot()[0]

    # Get headline vector and sentence vectors_sim0.0 from document
    for element in doc_root:
        if element.attrib['name'] == 'docno':
            doc_id = element.text
        elif element.attrib['name'] == 'headline':
            headline_vector = parse_headline(element.text)
        elif element.attrib['name'] == 'doctext':
            document_vectors = parse_document(element.text)

    vectors = get_relevant_vectors(headline_vector, document_vectors)

    # Pickle the objects so we can load the index index without these computations
    if not os.path.exists("vectors_sim{}".format(SIM_THRESHOLD)):
        os.makedirs("vectors_sim{}".format(SIM_THRESHOLD))
    pickle.dump(vectors, open("vectors_sim{}/{}.p".format(SIM_THRESHOLD, doc_id), "wb"))


def parse_headline(headline):
    return np.array(sentence_model([headline])[0].numpy())


def parse_document(doc_text):
    tokenized_sentences = tokenize_sentences(doc_text)
    return sentence_model(tokenized_sentences)


def get_relevant_vectors(headline, document_vectors):
    relevant_vectors = list()
    relevant_vectors.append(headline)

    # Add a sentence to relevant_vectors if it meets the similarity threshold to headline
    for sentence in document_vectors:
        vector = np.array(sentence.numpy())
        if euclidean_similarity(headline, vector) > SIM_THRESHOLD:
            relevant_vectors.append(vector)

    return relevant_vectors


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


for threshold in thresholds:
    SIM_THRESHOLD = threshold
    # Index each document in the data directory
    for i, file in enumerate(os.listdir(DATA_DIR)):
        if i % 1000 == 0:
            print(f"Parsed {i} documents")
        file_path = os.path.join(DATA_DIR, file)
        index_document(file_path)
