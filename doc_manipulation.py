from faker import Faker
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import string

def generate_doc_list():
    """
    This function generates a synthetic document with faker
    """
    fake = Faker()
    Faker.seed(0)
    fake_words = ['apple', 'car', 'dog', 'cat', 'house', 'tree', 'flower', 'book', 'computer']
    doc1 = ' '.join(fake.sentence(ext_word_list=fake_words) for _ in range(1))
    # Remove punctuators from doc1 and convert to lowercase
    doc1 = doc1.translate(str.maketrans('', '', string.punctuation))
    doc1 = doc1.lower()
    # Generate a similar document by replacing some words
    doc2 = doc1.replace('cat', 'car', 1)
    return [doc1, doc2]

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
    rw = np.random.choice([-1, 1], size=(dt_matrix.shape[1], m)) # random lines [term, m]
    simhash = np.zeros((dt_matrix.shape[0], m)) # initialize signature matrix [doc, m]
    for dt in dt_matrix:
        for i, w in enumerate(dt):
            simhash += w * rw[i]
    print(simhash)
    simhash = np.where(simhash > 0, 1, 0) # binarize signature matrix
    return simhash
