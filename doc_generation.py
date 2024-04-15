from faker import Faker

def doc_generate():
    """
    This function generates a synthetic document with faker
    """
    fake = Faker()
    document1 = ' '.join(fake.text() for _ in range(100))  # 100 sentences
    # Generate a similar document by replacing some words
    document2 = document1.replace('apple', 'fruit').replace('car', 'vehicle')
