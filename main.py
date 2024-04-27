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
    """
    Map Phase:
    Input: Your input data, which could be documents, images, etc.
    Processing: Compute the SimHash for each piece of input data. This could involve preprocessing the data, extracting features, and then hashing the features to compute the SimHash.
    Output: The output of the Map phase would be key-value pairs where the key is the SimHash and the value is the original data or some identifier for the data.
    Reduce Phase:
    Input: The key-value pairs output by the Map phase.
    Processing: For each set of values that share the same SimHash (i.e., they are potential duplicates), perform some operation. This could be as simple as listing the potential duplicates together, or it could involve further analysis.
    Output: The output of the Reduce phase would be the results of your duplicate detection or analysis.
    """
    spark = SparkSession.builder.appName('SimHash').getOrCreate()
    docs = dm.generate_synthetic_doc_list(spark)
    #docs = dm.generate_doc_list(spark)
    print(docs.take(10))
    comp_time = time.perf_counter()
    docs, n_terms = dm.compute_tfidf(docs.toDF(['index', 'text']))
    comp_time = time.perf_counter() - comp_time
    print('\ntfidf time', comp_time, '\n')
    print(docs.take(10))
    m = 64
    p = 8
    comp_time = time.perf_counter()
    rw = dm.compute_rw(spark, n_terms, m)
    comp_time = time.perf_counter() - comp_time
    print('\nrw time', comp_time, '\n')
    print(rw.take(10))
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
    groups = dm.gather_similar_simhash_groups(spark, simhash_pieces, p)
    comp_time = time.perf_counter() - comp_time
    print('\ngather time', comp_time, '\n')
    print(groups.take(10))



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

