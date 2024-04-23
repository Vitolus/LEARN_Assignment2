from dotenv import load_dotenv
import os
import doc_manipulation as dm
import time
import gc
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover


def local_main():
    #docs = dm.generate_doc_list()
    docs = dm.generate_synthetic_doc_list()
    print(docs)
    dt_matrix, terms = dm.compute_doc_term_matrix(docs)
    del docs
    gc.collect()
    m = 256
    rw = dm.compute_rw(terms.shape[0], m)
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(dt_matrix, rw)
    comp_time = time.perf_counter() - comp_time
    print('\nsimhash time', comp_time, '\n')
    p = m/8
    simhash_pieces = dm.split_simhash(simhash, p)
    simhash_ints = dm.pieces_to_ints(simhash_pieces)
    comp_time = time.perf_counter()
    hamming_distances = dm.compute_hamming_distances(simhash_ints)
    comp_time = time.perf_counter() - comp_time
    print('\nhamming time', comp_time, '\n')
    comp_time = time.perf_counter()
    cosine_similarities = dm.compute_cosine_similarities(hamming_distances, m)
    comp_time = time.perf_counter() - comp_time
    print('\ncosine time', comp_time, '\n')


def spark_main():
    spark = SparkSession.builder.appName('SimHash').getOrCreate()
    sc = spark.sparkContext
    docs = dm.generate_synthetic_doc_list(spark)
    #docs = dm.generate_doc_list(spark)
    print(docs.show(10))



if __name__ == "__main__":
    load_dotenv()
    os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
    pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
    if pyspark_python:
        print(f"PySpark is using this Python interpreter: {pyspark_python}")
    else:
        print("PySpark is using the system's default Python interpreter.")
    #local_main()
    spark_main()

