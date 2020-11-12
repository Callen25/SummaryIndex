import os
import xml.etree.ElementTree as ElementTree
from nltk.tokenize import sent_tokenize
import tensorflow_hub as hub
import cupy as np
from cupy import dot
from cupy.linalg import norm
import pickle

SIM_THRESHOLD = 0.25
DATA_DIR = "data"
sentence_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# def cosine_similarity(a, b):
#     return np.dot(a, b)/(norm(a)*norm(b))


def index_document(document):
    doc_root = ElementTree.parse(document).getroot()[0]

    # Get headline vector and sentence vectors from document
    for element in doc_root:
        if element.attrib['name'] == 'docno':
            doc_id = element.text
            print("doc_id: {}".format(doc_id))
        elif element.attrib['name'] == 'headline':
            headline_vector = parse_headline(element.text)
        elif element.attrib['name'] == 'doctext':
            document_vectors = parse_document(element.text)

    vectors = headline_vector + document_vectors  # get_relevant_vectors(headline_vector, document_vectors)

    # Pickle the objects so we can load the index index without these computations
    pickle.dump(vectors, open("vectors/{}.p".format(doc_id), "wb"))


def parse_headline(headline):
    return sentence_model([headline])[0]


def parse_document(doc_text):
    tokenized_sentences = tokenize_sentences(doc_text)
    return sentence_model(tokenized_sentences)


# def get_relevant_vectors(headline, document_vectors):
#     relevant_vectors = list()
#     relevant_vectors.append(headline)
#
#     # Add a sentence to relevant_vectors if it meets the similarity threshold to headline
#     for sentence in document_vectors:
#         if cosine_similarity(headline, sentence) > SIM_THRESHOLD:
#             relevant_vectors.append(sentence)
#
#     return relevant_vectors


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


# Index each document in the data directory
for file in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file)
    index_document(file_path)
