from dotenv import load_dotenv
import os
import local_doc_manipulation as ldm
import doc_manipulation as dm
import time
import gc
from pyspark.sql import SparkSession


def local_main():
    #docs = ldm.generate_doc_list()
    docs = ldm.generate_synthetic_doc_list()
    print(docs)
    dt_matrix, terms = ldm.compute_tfidf_terms(docs)
    del docs
    gc.collect()
    m = 256
    rw = ldm.compute_rw(terms.shape[0], m)
    comp_time = time.perf_counter()
    simhash = ldm.compute_simhash(dt_matrix, rw)
    comp_time = time.perf_counter() - comp_time
    print('\nsimhash time', comp_time, '\n')
    p = m/8
    simhash_pieces = ldm.split_simhash(simhash, p)
    simhash_ints = ldm.pieces_to_ints(simhash_pieces)
    comp_time = time.perf_counter()
    hamming_distances = ldm.compute_hamming_distances(simhash_ints)
    comp_time = time.perf_counter() - comp_time
    print('\nhamming time', comp_time, '\n')
    comp_time = time.perf_counter()
    cosine_similarities = ldm.compute_cosine_similarities(hamming_distances, m)
    comp_time = time.perf_counter() - comp_time
    print('\ncosine time', comp_time, '\n')


def spark_main():
    spark = SparkSession.builder.appName('SimHash').getOrCreate()
    docs = dm.generate_synthetic_doc_list(spark)
    #docs = dm.generate_doc_list(spark)
    docs.show(10)
    docs, n_terms = dm.compute_tfidf(docs)
    docs.show(10, truncate=False)
    m = 64
    p = 8
    rw = dm.compute_rw(spark, n_terms, m)
    rw.show(10, truncate=False)
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(spark, docs, rw, m)
    comp_time = time.perf_counter() - comp_time
    print('\nsimhash time', comp_time, '\n')
    simhash.show(10, truncate=False)
    comp_time = time.perf_counter()
    simhash_pieces = dm.split_simhash(spark, simhash, p)
    comp_time = time.perf_counter() - comp_time
    print('\nsplit time', comp_time, '\n')
    simhash_pieces.show(10, truncate=False)



if __name__ == "__main__":
    load_dotenv()
    os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
    pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
    if pyspark_python:
        print(f"PySpark is using this Python interpreter: {pyspark_python}")
    else:
        print("PySpark is using the system's default Python interpreter.")
    spark_main()
    # local_main()

