from dotenv import load_dotenv
import os
import local_doc_manipulation as ldm
import doc_manipulation as dm
import time
import gc
from pyspark.sql import SparkSession


def mapper(spark, docs, m, p, s):
    """
    Map Phase:
    Input: document corpus, simhash size m, number of pieces p, minimum similarity threshold s
    Processing: Compute the SimHash for each piece of input data. This could involve preprocessing the data, extracting
    features, and then hashing the features to compute the SimHash.
    Output: The output of the Map phase would be key-value pairs where the key is the SimHash and the value is the
    original data or some identifier for the data.
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
    groups = dm.gather_similar_simhash(spark, simhash_pieces, p)
    comp_time = time.perf_counter() - comp_time
    print('\ngather time', comp_time, '\n')
    print(groups.take(10))


def spark_main(m=64, p=8, s=0.9):
    """
    Reduce Phase:
    Input: The key-value pairs output by the Map phase.
    Processing: For each set of values that share the same SimHash (i.e., they are potential duplicates), perform some operation. This could be as simple as listing the potential duplicates together, or it could involve further analysis.
    Output: The output of the Reduce phase would be the results of your duplicate detection or analysis.
    """
    spark = SparkSession.builder.appName('SimHash').getOrCreate()
    docs = dm.generate_synthetic_doc_list(spark)
    #docs = dm.generate_doc_list(spark)
    print(docs.take(10))
    mapper(spark, docs, m, p, s)



if __name__ == "__main__":
    load_dotenv()
    os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
    pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
    if pyspark_python:
        print(f"PySpark is using this Python interpreter: {pyspark_python}")
    else:
        print("PySpark is using the system's default Python interpreter.")
    spark_main()

