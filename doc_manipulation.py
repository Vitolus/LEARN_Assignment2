from faker import Faker
import numpy as np
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, Normalizer
from pyspark.sql import functions as F
from pyspark.sql import Row, Window
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from operator import add
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
    docs = spark.sparkContext.parallelize(docs)
    return docs.zipWithIndex().map(lambda x: (x[1], x[0]))


def generate_doc_list(spark):
    # spark dataframe already parallelized (distributed)
    docs = spark.read.parquet("./data/train-00000-of-00004.parquet").withColumn('index', F.lit(0))  # ./data/*
    return docs.withColumn("index", F.row_number().over(Window.partitionBy('index').orderBy(F.lit(0))) - 1)


def compute_tfidf(docs):
    """
    This function computes the TF-IDF of a list of documents
    """
    docs = (StopWordsRemover(inputCol="words", outputCol="filtered")  # Remove stop words
            .transform(Tokenizer(inputCol="text", outputCol="words")  # Tokenize the documents
                       .transform(docs.withColumn("text", F.lower(F.regexp_replace(F.col("text"), '[^\w\s]', '')))))
            ).drop("text", "words")
    # Compute the number of distinct terms across all documents
    n_terms = docs.select(F.explode(docs.filtered)).distinct().count()
    docs = (CountVectorizer(inputCol="filtered", outputCol="tf")
            .fit(docs)
            .transform(docs)).drop("filtered")  # Compute the term frequencies
    docs = (IDF(inputCol="tf", outputCol="features")
            .fit(docs)
            .transform(docs)).drop("tf")  # Compute the inverse document frequencies
    return (Normalizer(inputCol="features", outputCol="tfidf", p=2.0)
            .transform(docs).drop("features").rdd.map(lambda row: (row['index'], row['tfidf'])),
            n_terms)  # Normalize the TF-IDF vectors


def compute_rw(spark, n_terms, m):
    """
    This function computes the random lines
    """
    # random lines [n_term, m]
    rw = map(lambda x: (Vectors.dense(x),), np.random.choice([-1, 1], size=(n_terms, m)))
    rw = spark.sparkContext.parallelize(rw)
    return rw.zipWithIndex().map(lambda x: (x[1], x[0]))


def compute_simhash(spark, docs, rw):
    """
    This function computes the simhash of the documents
    """
    # Broadcast the random lines for efficiency
    rw_broadcast = spark.sparkContext.broadcast(rw.collectAsMap())

    def simhash(doc_index, tfidf):
        # Initialize an empty signature vector
        signature = [0.0] * len(rw_broadcast.value[0])
        # For each word in the document
        for i in range(len(tfidf.indices)):
            # Get the word index and the TF-IDF value
            word_index = tfidf.indices[i]
            tfidf_value = tfidf.values[i]
            # Get the corresponding random line
            random_line = rw_broadcast.value[word_index]
            # Add the random line to the signature, scaled by the TF-IDF value
            scaled_line = [value * tfidf_value for value in random_line]
            signature = list(map(add, signature, scaled_line))
        # Convert the signature to a SimHash by taking the sign of each element
        simhash_bin = [1 if value > 0 else 0 for value in signature[0]]
        return doc_index, simhash_bin

    # Apply the SimHash function to each document
    simhash_rdd = docs.map(lambda x: simhash(*x))
    return simhash_rdd


def split_simhash(spark, simhash, p):
    """
    This function splits the simhash in p pieces
    """

    def split(doc_index, simhash):
        # Split the simhash into p pieces
        simhash_pieces = [simhash[i:i + p] for i in range(0, len(simhash), p)]
        # Convert each piece to an integer
        simhash_pieces = [int(''.join(str(i) for i in piece), 2) for piece in simhash_pieces]
        return doc_index, simhash_pieces

    # Apply the split_simhash function to each element of the RDD
    return simhash.map(lambda x: split(*x))


def compute_hamming_distance_piece(piece1, piece2):
    """
    This function computes the hamming distance between two int pieces of a simhash
    """
    hamm = bin(int(piece1) ^ int(piece2)).count('1')
    return hamm


def gather_similar_simhash_groups(spark, simhash, p):
    """
    This function gathers the similar simhash pairs for each document
    """
