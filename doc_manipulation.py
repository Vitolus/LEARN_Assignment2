from faker import Faker
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from math import cos


def generate_synthetic_doc_list():
    """
    This function generates synthetic documents with faker
    """
    fake = Faker()
    Faker.seed(0)
    docs = list()
    for _ in range(1):
        docs.append(fake.sentence(nb_words=10, ext_word_list=['ciao', 'cane', 'gatto'])
                    .translate(str.maketrans('', '', punctuation)).lower())
        docs.append(docs[0])
    docs.append(docs[0].replace('cane', 'gatto', 1))
    docs.append(docs[0].replace('cane', 'sasso', 3))
    return docs


def generate_doc_list():
    dataset = load_dataset("jacquelinehe/enron-emails")
    docs = np.empty(10000, dtype=object)
    for i in range(10000):
        docs[i] = dataset['train'][i]['text']
    return docs


def compute_doc_term_matrix(docs):
    """
    This function computes the document-term matrix
    """
    tfidf = TfidfVectorizer(stop_words='english')
    dt_matrix = tfidf.fit_transform(docs)
    print(dt_matrix.shape)
    terms = np.array(tfidf.get_feature_names_out())
    print(terms)
    print(dt_matrix)
    return dt_matrix, terms


def compute_simhash(dt_matrix, n_docs, n_terms, m):
    """
    This function computes the simhash of the documents
    """
    rw = np.random.choice([-1, 1], size=(n_terms, m))  # random lines [n_term, m]
    simhash = np.zeros((n_docs, m))  # initialize signature matrix [n_doc, m]
    # dt_matrix has shape [[(doc, term), tfidf], ...]
    print(n_docs, n_terms, dt_matrix.shape, rw.shape, simhash.shape)
    for i in range(n_docs):
        for j in range(n_terms):
            simhash[i] += dt_matrix[i, j] * rw[j]
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
    simhash_ints = np.empty((simhash_pieces.shape[0], simhash_pieces.shape[1]))
    for i, doc in enumerate(simhash_pieces):
        for j, piece in enumerate(doc):
            simhash_ints[i][j] = int(''.join(map(str, piece)), 2)
    print(simhash_ints)
    return simhash_ints


def compute_hamming_distance_piece(piece1, piece2):
    """
    This function computes the hamming distance between the simhashes
    """
    hamm = bin(np.bitwise_not(np.bitwise_xor(int(piece1), int(piece2)))).count('1') - 1
    return hamm


def compute_hamming_distances(simhash_ints):
    """
    This function computes the hamming distances between the simhashes
    """
    hamming_distances = list()
    for i in range(simhash_ints.shape[0] - 1):
        pieces = list()
        for j in range(i + 1, simhash_ints.shape[0]):
            pieces.append(np.array([compute_hamming_distance_piece((simhash_ints[i][k]), (simhash_ints[j][k]))
                                    for k in range(simhash_ints.shape[1])]))
        hamming_distances.append(np.array(pieces))
    print(hamming_distances)
    return hamming_distances


def compute_cosine_similarities(hamming_distances, m):
    """
    This function computes the cosine similarity between two documents
    """
    cosine_similarities = list()
    for doc in hamming_distances:
        cosines = np.empty((doc.shape[0]))
        for j, pieces in enumerate(doc):
            cosines[j] = (cos(sum(pieces) / m))
        cosine_similarities.append(cosines)
    print(cosine_similarities)
    return cosine_similarities
