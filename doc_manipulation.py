from faker import Faker
import pandas as pd
import numpy as np
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.sql.functions import explode, lower, regexp_replace, col, udf
from pyspark.sql import Row
from pyspark.ml.linalg import SparseVector, Vectors
import gc


def generate_synthetic_doc_list(spark):
    """
    This function generates synthetic documents with faker
    """
    fake = Faker()
    Faker.seed(0)
    docs = list()
    for _ in range(1):
        docs.append(fake.sentence(nb_words=15, ext_word_list=['hello', 'dog', 'cat']))
        docs.append(docs[0])
        docs.append(docs[0].replace('dog', 'cat', 1))
        docs.append(docs[0].replace('dog', 'cat', 3))
        docs.append('the pen is on the dusted table, but the table is not clean')
        docs.append('the pen is on the dusted floor, but the table is not clean')
    return spark.createDataFrame(pd.DataFrame(docs, columns=['text']))


def generate_doc_list(spark):
    # spark dataframe already parallelized (distributed)
    return spark.read.parquet("./data/train-00000-of-00004.parquet")  # ./data/*


def compute_tfidf(docs):
    """
    This function computes the TF-IDF of a list of documents
    """
    docs = (StopWordsRemover(inputCol="words", outputCol="filtered")  # Remove stop words
            .transform(Tokenizer(inputCol="text", outputCol="words")  # Tokenize the documents
                       .transform(docs.withColumn("text", lower(regexp_replace(col("text"), '[^\w\s]', ''))))))
    docs = docs.drop("text", "words")
    # Compute the number of distinct terms across all documents
    n_terms = docs.select(explode(docs.filtered)).distinct().count()
    docs = (HashingTF(inputCol="filtered", outputCol="tf")
            .transform(docs))  # Compute the term frequencies
    docs = docs.drop("filtered")
    docs.persist()  # Cache the dataframe
    return (IDF(inputCol="tf", outputCol="tfidf")
            .fit(docs)
            .transform(docs)).drop("tf"), n_terms  # Compute the inverse document frequencies


def compute_rw(spark, n_terms, m):
    """
    This function computes the random lines
    """
    # random lines [n_term, m]
    return spark.createDataFrame(map(lambda x: Row(rw=Vectors.dense(x)),
                                     np.random.choice([-1, 1], size=(n_terms, m))), ['rw'])


def compute_simhash(spark, docs, rw):
    """
    This function computes the simhash of the documents
    """

