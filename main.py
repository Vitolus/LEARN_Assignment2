from dotenv import load_dotenv
import os
import doc_manipulation as dm
import time
import numpy as np
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
    #docs, n_terms = dm.compute_tfidf(docs.toDF(['index', 'text']))
    docs, n_terms = dm.compute_tfidf(docs)
    comp_time = time.perf_counter() - comp_time
    print('\ntfidf time', comp_time, '\n')
    print(docs.take(10))
    comp_time = time.perf_counter()
    rw = dm.compute_rw(spark, n_terms, m)
    comp_time = time.perf_counter() - comp_time
    print('\nrw time', comp_time, '\n')
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(docs, rw)
    comp_time = time.perf_counter() - comp_time
    print('\nsimhash time', comp_time, '\n')
    comp_time = time.perf_counter()
    simhash_pieces = dm.split_simhash(simhash, p)
    comp_time = time.perf_counter() - comp_time
    print('\nsplit time', comp_time, '\n')
    print(simhash_pieces.take(10))

    def map_func(doc):
        doc_id, pieces = doc  # doc is a tuple (doc_id, pieces)
        pieces = np.sort(np.array(pieces))[::-1]  # sort the pieces in descending order
        pieces = pieces[:len(pieces)//2]  # take the first half of the pieces
        for piece in pieces:  # for each piece in the first half of the pieces
            yield piece, doc_id  # emit (piece, doc_id)

    return simhash_pieces.flatMap(lambda x: map_func(x)), simhash_pieces


def reducer(mapped, simhash_pieces, m, s):
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
    # flip the key-value pairs
    mapped = mapped.map(lambda x: (x[1], x[0]))
    # Join on docID
    joined = mapped.join(simhash_pieces)
    # Group by piece
    grouped = joined.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey()

    def reduce_func(key, values):
        for doc_id1, doc1 in values:
            for doc_id2, doc2 in values:
                doc1 = np.array(doc1)
                doc2 = np.array(doc2)
                if doc_id1 >= doc_id2 or key != max(np.intersect1d(doc1, doc2)):
                    continue
                shared_pieces = dm.count_shared_pieces(doc1, doc2)
                if shared_pieces >= len(doc1) // 2:
                    hamming = dm.compute_hamming_distance(doc1, doc2)
                    similarity = dm.compute_cosine_similarity(hamming, m)
                    if similarity >= s:
                        yield (doc_id1, doc_id2), similarity

    return grouped.flatMap(lambda x: reduce_func(x[0], x[1]))


def spark_main(m=64, p=8, s=0.95):
    spark = SparkSession.builder.appName('SimHash').getOrCreate()
    # docs = dm.generate_synthetic_doc_list(spark)
    # print(docs.take(10))
    docs = dm.generate_doc_list(spark)
    docs.show(10)
    comp_time = time.perf_counter()
    mapped, simhash_pieces = mapper(spark, docs, m, p)
    comp_time = time.perf_counter() - comp_time
    print('\nmap phase time', comp_time, '\n')
    print(mapped.take(10))
    comp_time = time.perf_counter()
    reduced = reducer(mapped, simhash_pieces, m, s)
    comp_time = time.perf_counter() - comp_time
    print('\nreduce phase time', comp_time, '\n')
    print(reduced.take(10))
    print('\nnumber of similar pairs', reduced.count())
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

