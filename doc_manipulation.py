from faker import Faker
import numpy as np
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, Normalizer
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors
import math
import gc


def generate_synthetic_doc_list(spark):
    """
    This function generates synthetic documents with faker
    """
    fake = Faker()
    Faker.seed(0)
    docs = list()
    for _ in range(1):
        docs.append(fake.sentence(nb_words=15, ext_word_list=['hello', 'dog', 'cat']))  # 0 doc
        docs.append(docs[0])  # 1 doc
        docs.append(docs[0].replace('dog', 'cat', 1))  # 2 doc
        docs.append(docs[0].replace('dog', 'cat', 3))  # 3 doc
        docs.append('the pen is on the dusted table, but the table is not clean')  # 4 doc
        docs.append('the pen is on the dusted floor, but the table is not clean')  # 5 doc
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
    docs = (CountVectorizer(inputCol="filtered", outputCol="tf").fit(docs).transform(docs)).drop("filtered")
    # Compute the inverse document frequencies
    docs = (IDF(inputCol="tf", outputCol="features").fit(docs).transform(docs)).drop("tf")
    # Normalize the TF-IDF vectors and return the result as an RDD
    return (Normalizer(inputCol="features", outputCol="tfidf", p=2.0).transform(docs).drop("features").rdd
            .map(lambda row: (row['index'], row['tfidf']))), n_terms  # map to (docID, tfidf)


def compute_rw(spark, n_terms, m):
    """
    This function computes the random lines
    """
    # random lines [n_term, m]
    np.random.seed(0)
    rw = [(Vectors.dense(x),) for x in np.random.choice([-1, 1], size=(n_terms, m))]  # random lines [n_term, m]
    rw = spark.sparkContext.parallelize(rw)
    return rw.zipWithIndex().map(lambda x: (x[1], x[0]))  # map to (termID, random_line)


def compute_simhash(docs, rw):
    """
    This function computes the simhash of the documents
    """
    # Explode the documents to get a (docID, (termID, tfidf_value)) RDD
    exploded_docs = docs.flatMap(lambda x: ((int(term_id), (x[0], x[1][int(term_id)])) for term_id in x[1].indices))
    # Join the exploded_docs and rw RDDs based on termID and group by docID
    grouped_rdd = exploded_docs.join(rw).groupBy(lambda x: x[1][0][0])

    def simhash(doc_index, tfidf_values, random_lines):
        # Initialize an empty signature vector
        signature = [0.0] * len(random_lines[0][0])
        # For each word in the document
        for tfidf_value, random_line in zip(tfidf_values, random_lines):
            # Add the random line to the signature, scaled by the TF-IDF value
            signature = [s + value * tfidf_value for s, value in zip(signature, random_line[0])]
        # Convert the signature to a SimHash by taking the sign of each element
        simhash_bin = [1 if value > 0 else 0 for value in signature]
        return doc_index, simhash_bin

    # Apply the SimHash function to each group
    return grouped_rdd.map(
        lambda x: simhash(x[0], (value[1][0][1] for value in list(x[1])), [value[1][1] for value in list(x[1])]))


def split_simhash(simhash, p):
    """
    This function splits the simhash in p pieces
    """
    # Apply the split_simhash function to each element of the RDD
    simhash_pieces = simhash.map(lambda x: split(*x))

    def split(doc_index, sims):
        # Split the simhash into p pieces
        sim_pieces = [sims[i:i + p] for i in range(0, len(sims), p)]
        return doc_index, sim_pieces

    # Convert each piece to an integer
    return simhash_pieces.map(lambda x: (x[0], [int(''.join(str(bit) for bit in piece), 2) for piece in x[1]]))


def count_shared_pieces(doc1, doc2):
    """
    This function counts the number of shared pieces between two documents
    """
    return np.count_nonzero(doc1 == doc2)


def compute_hamming_distance(doc1, doc2):
    """
    This function computes the hamming distance between two documents
    """
    return np.sum(np.vectorize(lambda x: (bin(x).count('1')))(np.bitwise_xor(doc1, doc2)))


def compute_cosine_similarity(hamming, m):
    """
    This function computes the cosine similarity between two documents
    """
    return math.cos(hamming / m)
