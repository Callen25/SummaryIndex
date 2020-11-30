import os
import xml.etree.ElementTree as ElementTree
from nltk.tokenize import sent_tokenize
import tensorflow_hub as hub
import cupy as np
from cupy.linalg import norm
import pickle

SIM_THRESHOLD = 0.0
DATA_DIR = "data"
sentence_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def cosine_similarity(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))


def euclidean_distance(a, b):
    distance = norm(a - b)
    return 1 / (1 + distance)


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
        if cosine_similarity(headline, vector) > SIM_THRESHOLD:
            relevant_vectors.append(vector)

    return relevant_vectors


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


# Index each document in the data directory
for i, file in enumerate(os.listdir(DATA_DIR)):
    if i % 1000 == 0:
        print(f"Parsed {i} documents")
    file_path = os.path.join(DATA_DIR, file)
    index_document(file_path)
