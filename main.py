import sys
import doc_manipulation as dm
import time
import numpy as np
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import os


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
    tfidf, n_terms = dm.compute_tfidf(docs)
    docs.unpersist()
    print('\ntfidf time', time.perf_counter() - comp_time, '\n')
    comp_time = time.perf_counter()
    rw = dm.compute_rw(spark, n_terms, m)
    print('\nrw time', time.perf_counter() - comp_time, '\n')
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(tfidf, rw)
    print('\nsimhash time', time.perf_counter() - comp_time, '\n')
    comp_time = time.perf_counter()
    simhash_pieces = dm.split_simhash(simhash, p)
    print('\nsplit time', time.perf_counter() - comp_time, '\n')
    print(simhash_pieces.take(10))

    def map_func(doc):
        doc_id, pieces = doc  # doc is a tuple (doc_id, pieces)
        pieces = np.sort(pieces)  # sort the pieces in ascending order
        # take the first half of the pieces to reduce the number of copies across nodes,
        # at the cost of some accuracy in the number of similar pairs
        pieces = pieces[:len(pieces) // 2]
        for piece in pieces:
            yield piece, doc_id

    return simhash_pieces.flatMap(map_func), simhash_pieces  # return the mapped RDD and the simhash pieces


def reducer(mapped, simhash_pieces, m, s):
    """
    Reduce Phase:
    - Group by Piece: Group the documents that have at least half of the pieces of the SimHash equal.
    - Compute Hamming Distance: For each group of documents (that share at least half of the SimHash pieces), compute
    the Hamming distance between the SimHashes of each pair of documents.
    - Compute Cosine Similarity: Still within each group, compute the cosine similarity for each pair of documents.
    Emit Similar Documents: Finally, for each document, emit the list of documents that are considered similar based on
    the computed cosine similarity and Hamming distance.
    """
    flipped = mapped.map(lambda x: (x[1], x[0]))  # flip the key-value pairs
    joined = flipped.join(simhash_pieces)  # Join on docID
    # Group by piece and sort by docID in each group
    grouped = joined.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().mapValues(list)

    def reduce_func(key, values):
        values = sorted(values, key=lambda x: x[0])  # sort the values by doc_id
        for i in range(len(values)):
            doc_id1, doc1 = values[i]
            doc1 = np.array(doc1)
            for j in range(i + 1, len(values)):
                doc_id2, doc2 = values[j]
                doc2 = np.array(doc2)
                # if the key is not the minimum of the intersection of doc1 and doc2 skip this iteration, this
                # guarantees that two documents are only compared once
                if key != min(np.intersect1d(doc1, doc2)):
                    continue
                # count the number of shared pieces between doc1 and doc2
                shared_pieces = dm.count_shared_pieces(doc1, doc2)
                # if the number of shared pieces is greater than or equal to half of the length of doc1
                if shared_pieces >= len(doc1) // 2:
                    # compute the hamming distance between doc1 and doc2
                    hamming = dm.compute_hamming_distance(doc1, doc2)
                    # if the hamming distance is less than or equal to s bits and greater than 0 to exclude the pair of
                    # documents that are equal
                    if hamming > 0:
                        # compute the cosine similarity between doc1 and doc2
                        similarity = dm.compute_cosine_similarity(hamming, m)
                        if similarity >= s:
                            yield (doc_id1, doc_id2), similarity

    return grouped.flatMap(lambda x: reduce_func(x[0], x[1]))  # return the reduced RDD


def spark_main(ext="parquet", path="./data/emails/*", m=64, p=8, s=0.9):
    spark = SparkSession.builder.appName('SimHash').getOrCreate()  # create a Spark session
    docs = dm.generate_doc_list(spark, ext, path).limit(10000)  # generate a list of documents
    docs.persist()
    print('\nnumber of documents:', docs.count())
    print('length of the signature:', m, 'bits')
    print('length of the pieces:', p, 'bits')
    print('similarity threshold:', s * 100, '%\n')
    docs.show(10)
    comp_time = time.perf_counter()
    mapped, simhash_pieces = mapper(spark, docs, m, p)  # map phase
    print(mapped.take(10))
    print('\nmap phase time', time.perf_counter() - comp_time, '\n')
    comp_time = time.perf_counter()
    reduced = reducer(mapped, simhash_pieces, m, s)  # reduce phase
    reduced.persist()
    print(reduced.take(10))
    print('\nreduce phase time', time.perf_counter() - comp_time, '\n')
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

    if len(sys.argv) == 1:
        spark_main()
    elif len(sys.argv) == 3:
        spark_main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 6:
        spark_main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
    else:
        print("Usage: spark-submit main.py <extension of file> <path> [<m> <p> <s>]")
        sys.exit(1)
