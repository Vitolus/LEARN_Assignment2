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
    - Compute SimHash: For each document, compute the SimHash. This can be done in parallel for all documents.
    - Split SimHash: Split the SimHash of each document into pieces of equal length. This operation is also independent
    for each document and can be done in parallel.
    - Emit Pairs: For each document, emit pairs of the form (piece, document_id). This will allow grouping by piece in
    the next phase.
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
    comp_time = time.perf_counter()
    simhash_pieces = dm.split_simhash(spark, simhash, p)
    comp_time = time.perf_counter() - comp_time
    print('\nsplit time', comp_time, '\n')
    print(simhash_pieces.take(10))

    def map_func(doc):
        doc_id, pieces = doc  # doc is a tuple (doc_id, pieces)
        for piece in pieces:  # piece is a list of p pieces
            yield piece, doc_id  # emit (piece, doc_id)

    return simhash_pieces.flatMap(map_func)


def reducer(spark, simhash_groups, s):
    """
    Reduce Phase:
    - Group by Piece: Group the documents that have at least half of the pieces of the SimHash equal. This is done by
    Spark under the hood when you call groupByKey() or similar operations.
    - Compute Hamming Distance: For each group of documents (that share at least half of the SimHash pieces), compute
    the Hamming distance between the SimHashes of each pair of documents.
    - Compute Cosine Similarity: Still within each group, compute the cosine similarity for each pair of documents.
    Emit Similar Documents: Finally, for each document, emit the list of documents that are considered similar based on
    the computed cosine similarity and Hamming distance.
    """



def spark_main(m=64, p=8, s=0.9):
    spark = SparkSession.builder.appName('SimHash').getOrCreate()
    docs = dm.generate_synthetic_doc_list(spark)
    #docs = dm.generate_doc_list(spark)
    print(docs.take(10))
    comp_time = time.perf_counter()
    mapped = mapper(spark, docs, m, p)
    comp_time = time.perf_counter() - comp_time
    print('\nmap phase time', comp_time, '\n')
    print(mapped.take(10))
    comp_time = time.perf_counter()
    reduced = reducer(spark, mapped, s)
    comp_time = time.perf_counter() - comp_time
    print('\nreduce phase time', comp_time, '\n')
    print(reduced.take(10))
    spark.stop()


if __name__ == "__main__":
    load_dotenv()
    os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
    pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
    if pyspark_python:
        print(f"PySpark is using this Python interpreter: {pyspark_python}")
    else:
        print("PySpark is using the system's default Python interpreter.")
    spark_main()

