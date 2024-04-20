from faker import Faker
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from math import cos


def generate_doc_list():
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
    docs.append(docs[0].replace('cane', 'gatto', 3))
    print(docs)
    return docs


def compute_doc_term_matrix(docs):
    """
    This function computes the document-term matrix
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(docs)
    terms = np.array(tfidf.get_feature_names_out())
    dt_matrix = tfidf.transform(docs).toarray()
    print(terms)
    print(dt_matrix)
    return dt_matrix, terms


def compute_simhash(dt_matrix, m):
    """
    This function computes the simhash of the documents
    """
    rw = np.random.choice([-1, 1], size=(dt_matrix.shape[1], m))  # random lines [term, m]
    simhash = np.zeros((dt_matrix.shape[0], m))  # initialize signature matrix [doc, m]
    for i, doc in enumerate(dt_matrix):
        for j, term in enumerate(doc):
            simhash[i] += term * rw[j]
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


def ints_to_pieces(simhash_ints):
    """
    This function converts each integer to piece of simhash
    """
    simhash_pieces = np.empty((simhash_ints.shape[0], simhash_ints.shape[1]), dtype=object)
    for i, doc in enumerate(simhash_ints):
        for j, num in enumerate(doc):
            simhash_pieces[i][j] = np.array(list(int(bit) for bit in bin(int(num))[2:].zfill(32)))
    print(simhash_pieces)
    return simhash_pieces


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
