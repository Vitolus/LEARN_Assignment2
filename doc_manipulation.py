from faker import Faker
import numpy as np
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, Normalizer
from pyspark.sql import functions as F
from pyspark.sql import Row, Window
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from operator import add
from itertools import islice
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
    # Tokenize the documents and remove stop words
    docs = (StopWordsRemover(inputCol="words", outputCol="filtered")
            .transform(Tokenizer(inputCol="text", outputCol="words")
                       .transform(docs.withColumn("text", F.lower(F.regexp_replace(F.col("text"), '[^\w\s]', '')))))
            ).drop("text", "words")
    docs.persist()
    # Compute the number of distinct terms across all documents
    n_terms = docs.select(F.explode(docs.filtered)).distinct().count()
    # Compute the term frequencies
    docs = (CountVectorizer(inputCol="filtered", outputCol="tf")
            .fit(docs)
            .transform(docs)).drop("filtered")
    # Compute the inverse document frequencies
    docs = (IDF(inputCol="tf", outputCol="features")
            .fit(docs)
            .transform(docs)).drop("tf")
    # Normalize the TF-IDF vectors and return the result as an RDD
    return (Normalizer(inputCol="features", outputCol="tfidf", p=2.0)
            .transform(docs).drop("features").rdd.map(lambda row: (row['index'], row['tfidf']))), n_terms  # Normalize the TF-IDF vectors


def compute_rw(spark, n_terms, m):
    """
    This function computes the random lines
    """
    # random lines [n_term, m]
    np.random.seed(0)
    rw = [(Vectors.dense(x),) for x in np.random.choice([-1, 1], size=(n_terms, m))]
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
            signature = [s + value * tfidf_value for s, value in zip(signature, random_line)]
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

    def split(doc_index, sims):
        # Split the simhash into p pieces
        sim_pieces = [sims[i:i + p] for i in range(0, len(sims), p)]
        return doc_index, sim_pieces

    # Apply the split_simhash function to each element of the RDD
    simhash_pieces = simhash.map(lambda x: split(*x))
    # Convert each piece to an integer
    return simhash_pieces.map(lambda x: (x[0], [int(''.join(str(bit) for bit in piece), 2) for piece in x[1]]))


def gather_similar_simhash(spark, simhash, p):
    """
    This function gathers groups of documents that share at least half of their simhash pieces
    """
    # Broadcast the simhash RDD
    simhash_broadcast = spark.sparkContext.broadcast(simhash.collectAsMap())

    def gather(doc_index, pieces):
        similar_docs = list()
        for succ_index in islice(simhash_broadcast.value, doc_index + 1, None):
            n_similar = sum(p1 == p2 for p1, p2 in zip(pieces, simhash_broadcast.value[succ_index]))
            if n_similar >= p / 2:
                similar_docs.append(succ_index)
        return doc_index, similar_docs

    return simhash.map(lambda x: gather(*x))


def compute_hamming_distance_piece(piece1, piece2):
    """
    This function computes the hamming distance between two int pieces of a simhash
    """
    hamm = bin(int(piece1) ^ int(piece2)).count('1')
    return hamm


