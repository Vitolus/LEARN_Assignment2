from dotenv import load_dotenv
import os
import local_doc_manipulation as ldm
import doc_manipulation as dm
import time
import gc
from pyspark.sql import SparkSession


def mapper(spark, docs, m, p):
    """
    Map Phase:
    Input: Document corpus, simhash size m, number of pieces p, minimum similarity threshold s
    Processing: Compute the SimHash for each input document. Split each SimHash into p pieces. Group the documents by
    potential similarity.
    Output: Key-value pairs where the key is the docID and the value is the SimHash. Key-value pairs where the key is
    the docID and the value is a list of docID of potentially similar documents
    """
    comp_time = time.perf_counter()
    docs, n_terms = dm.compute_tfidf(docs.toDF(['index', 'text']))
    comp_time = time.perf_counter() - comp_time
    print('\ntfidf time', comp_time, '\n')
    print(docs.take(10))
    comp_time = time.perf_counter()
    rw = dm.compute_rw(spark, n_terms, m)
    comp_time = time.perf_counter() - comp_time
    print('\nrw time', comp_time, '\n')
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(spark, docs, rw)
    comp_time = time.perf_counter() - comp_time
    print('\nsimhash time', comp_time, '\n')
    print(simhash.take(10))
    comp_time = time.perf_counter()
    simhash_pieces = dm.split_simhash(spark, simhash, p)
    comp_time = time.perf_counter() - comp_time
    print('\nsplit time', comp_time, '\n')
    print(simhash_pieces.take(10))
    comp_time = time.perf_counter()
    groups = dm.gather_similar_simhash(spark, simhash_pieces, p)
    comp_time = time.perf_counter() - comp_time
    print('\ngather time', comp_time, '\n')
    print(groups.take(10))
    return simhash_pieces, groups


def reducer(spark, simhash, groups, s):
    """
    Reduce Phase:
    Input: The 2 sets key-value pairs output by the Map phase.
    Processing: For each set of values that are potentially similar, compute the cosine similarity between the documents.
    Output: Key-value pairs where the key is the docID and the value is a list of docID of similar documents
    """
    comp_time = time.perf_counter()
    hamming_distances = dm.compute_hamming_distances(spark, simhash, groups)
    comp_time = time.perf_counter() - comp_time
    print('\nhamming time', comp_time, '\n')
    print(hamming_distances.take(10))


def spark_main(m=64, p=8, s=0.9):
    spark = SparkSession.builder.appName('SimHash').getOrCreate()
    docs = dm.generate_synthetic_doc_list(spark)
    #docs = dm.generate_doc_list(spark)
    print(docs.take(10))
    simhash, groups = mapper(spark, docs, m, p)
    #reducer(spark, simhash, groups, s)



if __name__ == "__main__":
    load_dotenv()
    os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
    pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
    if pyspark_python:
        print(f"PySpark is using this Python interpreter: {pyspark_python}")
    else:
        print("PySpark is using the system's default Python interpreter.")
    spark_main()

