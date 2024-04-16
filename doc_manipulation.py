from faker import Faker
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
