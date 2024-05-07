from faker import Faker
import numpy as np
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, Normalizer
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors
import math


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


def generate_doc_list(spark, path):
    docs = spark.read.parquet(path).withColumn('index', F.lit(0))
    return docs.withColumn("index", F.row_number().over(Window.partitionBy('index').orderBy(F.lit(0))) - 1)


def compute_tfidf(docs):
    """
    This function computes the TF-IDF of a list of documents
    """
    # Tokenize the documents and remove stop words
    token = (StopWordsRemover(inputCol="words", outputCol="filtered")
             .transform(Tokenizer(inputCol="text", outputCol="words")
                        .transform(docs.withColumn("text", F.lower(F.regexp_replace(F.col("text"), '[^\w\s]', '')))))
             ).drop("text", "words")
    # Compute the number of distinct terms across all documents
    n_terms = token.select(F.explode(token.filtered)).distinct().count()
    # Compute the term frequencies
    tf = (CountVectorizer(inputCol="filtered", outputCol="tf").fit(token).transform(token)).drop("filtered")
    # Compute the inverse document frequencies
    idf = (IDF(inputCol="tf", outputCol="features").fit(tf).transform(tf)).drop("tf")
    # Normalize the TF-IDF vectors and return the result as an RDD
    return (Normalizer(inputCol="features", outputCol="tfidf", p=2.0).transform(idf).drop("features").rdd
            .map(lambda row: (row['index'], row['tfidf']))), n_terms  # map to (docID, tfidf)


def compute_rw(spark, n_terms, m):
    """
    This function computes the random lines
    """
    np.random.seed(0)
    # Generate random lines of -1 and 1 with shape (n_terms, m)
    rw = spark.sparkContext.parallelize(np.random.choice([-1, 1], size=(n_terms, m)))
    # Convert to dense vectors and zip with index to get (wordID, random_line) pairs
    return rw.map(lambda x: (Vectors.dense(x),)).zipWithIndex().map(lambda x: (x[1], x[0]))


def compute_simhash(docs, rw):
    """
    This function computes the simhash of the documents
    """
    # Explode the documents to get a (docID, (termID, tfidf_value)) RDD
    exploded_docs = docs.flatMap(lambda x: ((int(term_id), (x[0], x[1][int(term_id)])) for term_id in x[1].indices))
    # Join the exploded_docs and rw RDDs based on termID and group by docID
    grouped_rdd = exploded_docs.join(rw).groupBy(lambda x: x[1][0][0])

    def simhash(doc_index, tfidf_values, random_lines):
        signature = np.zeros(len(random_lines[0][0]))  # Initialize an empty signature vector
        for tfidf_value, random_line in zip(tfidf_values, random_lines):  # For each word in the document
            # Add the random line to the signature, scaled by the TF-IDF value
            signature += np.array(random_line[0]) * tfidf_value
        # Convert the signature to a SimHash by taking the sign of each element
        simhash_bin = [1 if value > 0 else 0 for value in signature]
        yield doc_index, simhash_bin

    # Apply the SimHash function to each group
    return grouped_rdd.flatMap(
        lambda x: simhash(x[0], (value[1][0][1] for value in list(x[1])), [value[1][1] for value in list(x[1])]))


def split_simhash(simhash, p):
    """
    This function splits the simhash in p pieces
    """
    def split(doc_index, sims):
        # Split the simhash into p pieces
        sim_pieces = [sims[i:i + p] for i in range(0, len(sims), p)]
        # Convert each piece to an integer
        yield doc_index, [int(''.join(str(bit) for bit in piece), 2) for piece in sim_pieces]

    # Apply the split_simhash function to each element of the RDD
    return simhash.flatMap(lambda x: split(*x))


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
    return math.cos(hamming / m * math.pi / 2)
