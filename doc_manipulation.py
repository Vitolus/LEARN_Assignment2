from faker import Faker
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import string


def generate_doc_list():
    """
    This function generates synthetic documents with faker
    """
    fake = Faker()
    Faker.seed(0)
    docs = list()
    for _ in range(1):
        docs.append(fake.sentence(nb_words=10, ext_word_list=['ciao', 'cane', 'gatto'])
                    .translate(str.maketrans('', '', string.punctuation)).lower())
    docs.append(docs[0].replace('cane', 'gatto', 1))
    return docs


def compute_doc_term_matrix(docs):
    """
    This function computes the document-term matrix
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(docs)
    terms = np.array(tfidf.get_feature_names_out())
    dt_matrix = np.array(tfidf.transform(docs).toarray())
    return dt_matrix, terms


def compute_simhash(dt_matrix, m):
    """
    This function computes the simhash of the documents
    """
    rw = np.random.choice([-1, 1], size=(dt_matrix.shape[1], m))  # random lines [term, m]
    simhash = np.zeros((dt_matrix.shape[0], m))  # initialize signature matrix [doc, m]
    for i, dt in enumerate(dt_matrix):
        for j, w in enumerate(dt):
            simhash[i] += w * rw[j]
    print(simhash)
    return np.where(simhash > 0, 1, 0)  # binarize signature matrix


def split_simhash(simhash, p):
    """
    This function splits the simhash in p pieces
    """
    return np.array(np.array_split(simhash, p, axis=1))


def convert_split_to_int(simhash_pieces):
    """
    This function converts each piece of simhash to integer
    """
    print('-----------------------------')
    simhash_ints = np.empty((simhash_pieces.shape[0], simhash_pieces.shape[1]))
    for i, doc in enumerate(simhash_pieces):
        for j, piece in enumerate(doc):
            simhash_ints[i][j] = int(''.join(map(str, piece)), 2)
    return simhash_ints
