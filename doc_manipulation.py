from faker import Faker
import pandas as pd
from string import punctuation


def generate_synthetic_doc_list(spark):
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
    return spark.createDataFrame(pd.DataFrame(docs, columns=['text']))


def generate_doc_list(spark):
    # spark dataframe already parallelized (distributed)
    return spark.read.parquet("./data/train-00000-of-00004.parquet")  # ./data/*
