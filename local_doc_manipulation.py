from faker import Faker
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation


def generate_synthetic_doc_list():
    """
    This function generates synthetic documents with faker
    """
    fake = Faker()
    Faker.seed(0)
    docs = list()
    for _ in range(1):
        docs.append(fake.sentence(nb_words=15, ext_word_list=['hello', 'dog', 'cat'])
                    .translate(str.maketrans('', '', punctuation)).lower())
        docs.append(docs[0])
        docs.append(docs[0].replace('dog', 'cat', 1))
        docs.append(docs[0].replace('dog', 'cat', 3))
        docs.append('the pen is on the dusted table but the table is not clean')
        docs.append('the pen is on the dusted floor but the table is not clean')
    return pd.DataFrame(docs, columns=['text'])


def generate_doc_list(spark):
    # spark dataframe already parallelized (distributed)
    return spark.read.parquet("./data/train-00000-of-00004.parquet")  # ./data/*


def compute_tfidf_terms(docs):
    """
    This function computes the document-term matrix
    """
    tfidf = TfidfVectorizer(stop_words='english')
    dt_matrix = tfidf.fit_transform(docs)
    print('dt shape', dt_matrix.shape)
    terms = np.array(tfidf.get_feature_names_out())
    print('terms', terms)
    print('dtmatrix', dt_matrix)
    return dt_matrix, terms


def compute_rw(n_terms, m):
    """
    This function computes the random lines
    """
    rw = np.random.choice([-1, 1], size=(n_terms, m))  # random lines [n_term, m]
    return rw


def compute_simhash(dt_matrix, rw):
    """
    This function computes the simhash of the documents
    """
    simhash = dt_matrix.dot(rw)  # [n_doc, m]
    simhash = np.where(simhash > 0, 1, 0)  # binarize signature matrix
    return simhash


def split_simhash(simhash, p):
    """
    This function splits the simhash in p pieces
    """
    simhash_pieces = np.array([np.array_split(array, p, axis=0) for array in simhash])
    print(simhash_pieces)
    return simhash_pieces


def pieces_to_ints(simhash_pieces):
    """
    This function converts each piece of simhash to integer
    """
    simhash_ints = np.apply_along_axis(lambda x: int(''.join(map(str, x)), 2), axis=2, arr=simhash_pieces)
    print(simhash_ints)
    return simhash_ints


def compute_hamming_distance_piece(piece1, piece2):
    """
    This function computes the hamming distance between two int pieces of a simhash
    """
    hamm = bin(int(piece1) ^ int(piece2)).count('1')
    return hamm


def compute_hamming_distances(simhash_ints):
    """
    This function computes the hamming distances between the simhashes
    """
    n_docs, n_pieces = simhash_ints.shape
    hamming_distances = list()
    for i in range(n_docs - 1):
        pieces = [[compute_hamming_distance_piece(simhash_ints[i, k], simhash_ints[j, k])
                   for k in range(n_pieces)] for j in range(i + 1, n_docs)]
        hamming_distances.append(np.array(pieces))
    print(hamming_distances)
    return hamming_distances


def compute_cosine_similarities(hamming_distances, m):
    """
    This function computes the cosine similarity between two documents
    """
    cosine_similarities = [np.cos(np.sum(doc, axis=1) / m) for doc in hamming_distances]
    print(cosine_similarities)
    return cosine_similarities
