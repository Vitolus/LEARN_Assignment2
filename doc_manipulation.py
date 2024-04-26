from faker import Faker
import pandas as pd
import numpy as np
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, Normalizer
from pyspark.sql import functions as F
from pyspark.sql import Row, Window
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
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
    docs = spark.createDataFrame(pd.DataFrame(docs, columns=['text'])).withColumn('index', F.lit(0))
    return docs.withColumn("index", F.row_number().over(Window.partitionBy('index').orderBy(F.lit(0))) - 1)


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
            .transform(docs).drop("features"), n_terms)  # Normalize the TF-IDF vectors


def compute_rw(spark, n_terms, m):
    """
    This function computes the random lines
    """
    # random lines [n_term, m]
    rw = spark.createDataFrame(map(lambda x: Row(rw=Vectors.dense(x)),
                                   np.random.choice([-1, 1], size=(n_terms, m))), ['rw']).withColumn('index', F.lit(0))
    return rw.withColumn("index", F.row_number().over(Window.partitionBy('index').orderBy(F.lit(0))) - 1)


def compute_simhash(spark, docs, rw, m):
    """
    This function computes the simhash of the documents
    """

    # create a new empty dataframe to store simhash with columns index of integer and simhash of vector
    simhash = spark.createDataFrame([], StructType([StructField("index", IntegerType(), False),
                                                    StructField("simhash", ArrayType(IntegerType()), False)]))
    for doc in range(docs.count()):  # iterate over the documents
        tfidf_row = docs.filter(docs.index == doc).select('tfidf').collect()[0].tfidf  # get the tfidf of the document
        val = np.zeros(m)  # initialize the simhash vector
        for i in range(tfidf_row.indices.size):  # iterate over the non-zero elements of the tfidf vector
            val += tfidf_row.values[i] * rw.filter(rw.index == tfidf_row.indices[i]).select('rw').collect()[0].rw  # add the wighted random line to the simhash vector
        val = list(map(lambda y: 1 if y > 0 else 0, val))  # binarize the simhash vector
        simhash = simhash.union(spark.createDataFrame([Row(index=doc, sim=val)]))  # append the simhash vector to the dataframe
    # binarize signature matrix
    return simhash


def split_simhash(spark, simhash, p):
    """
    This function splits the simhash in p pieces
    """
    simhash = simhash.withColumn('pieces', F.udf(lambda lst: [lst[i:i + p] for i in range(0, len(lst), p)],
                                                 ArrayType(ArrayType(IntegerType())))(simhash['simhash']))
    for i in range(p):
        simhash = simhash.withColumn(f'piece{i + 1}', F.udf(lambda lst: int(''.join(str(i) for i in lst), 2),
                                                            IntegerType())(simhash['pieces'][i]))
    return simhash.drop('simhash', 'pieces')
