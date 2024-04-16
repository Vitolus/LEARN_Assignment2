from faker import Faker
import numpy as np
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

def compute_simhash(dt_matrix, terms):
    """
    This function computes the simhash of the documents
    """
    # Compute the simhash of the documents
    simhash = np.zeros((dt_matrix.shape[0], 64))
    # TODO: chack, seems broken
    for i, doc in enumerate(dt_matrix):
        for j, term in enumerate(terms):
            simhash[i] += (doc[j] > 0) * hash(term)
    return simhash
