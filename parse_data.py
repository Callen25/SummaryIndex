import os
import xml.etree.ElementTree as ElementTree
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import pickle


DATA_DIR = "data"
SENTENCE_TRANSFORMER = SentenceTransformer('bert-base-nli-mean-tokens')
SIM_THRESHOLD = 0.75


def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))


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

    vectors = get_relevant_vectors(headline_vector, document_vectors)

    # Pickle the objects so we can load the index index without these computations
    pickle.dump(vectors, open("vectors/{}.p".format(doc_id), "wb"))


def parse_headline(headline):
    tokenized_headline = list()
    tokenized_headline.append(word_tokenize(headline))
    return SENTENCE_TRANSFORMER.encode(tokenized_headline)[0]


def parse_document(doc_text):
    tokenized_sentences = tokenize_sentences(doc_text)
    return SENTENCE_TRANSFORMER.encode(tokenized_sentences)


def get_relevant_vectors(headline, document_vectors):
    relevant_vectors = list()
    relevant_vectors.append(headline)

    # Add a sentence to relevant_vectors if it meets the similarity threshold to headline
    for sentence in document_vectors:
        if cosine_similarity(headline, sentence) > SIM_THRESHOLD:
            relevant_vectors.append(sentence)

    return relevant_vectors


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    tokenized_sentences = []
    for sentence in sentences:
        if len(sentence) > 4:
            tokenized_sentences.append(word_tokenize(sentence.lower()))
    return tokenized_sentences


# Index each document in the data directory
for file in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file)
    index_document(file_path)
