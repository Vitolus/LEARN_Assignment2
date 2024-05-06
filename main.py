#from dotenv import load_dotenv
#import os
import sys
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
    docs, n_terms = dm.compute_tfidf(docs)
    print('\ntfidf time', time.perf_counter() - comp_time, '\n')
    print(docs.take(10))
    comp_time = time.perf_counter()
    rw = dm.compute_rw(spark, n_terms, m)
    print('\nrw time', time.perf_counter() - comp_time, '\n')
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(docs, rw)
    print('\nsimhash time', time.perf_counter() - comp_time, '\n')
    comp_time = time.perf_counter()
    simhash_pieces = dm.split_simhash(simhash, p)
    print('\nsplit time', time.perf_counter() - comp_time, '\n')
    print(simhash_pieces.take(10))

    def map_func(doc):
        doc_id, pieces = doc  # doc is a tuple (doc_id, pieces)
        pieces = np.sort(pieces)[::-1]  # sort the pieces in descending order
        # take the first half of the pieces to reduce the number of copies across nodes,
        # at the cost of some accuracy in the number of similar pairs
        pieces = pieces[:len(pieces)//2]
        for piece in pieces:  # for each piece in the first half of the pieces
            yield piece, doc_id

    return simhash_pieces.flatMap(map_func), simhash_pieces  # return the mapped RDD and the simhash pieces


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
    mapped = mapped.map(lambda x: (x[1], x[0]))  # flip the key-value pairs
    joined = mapped.join(simhash_pieces)  # Join on docID
    # Group by piece and sort by docID in each group
    grouped = joined.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().mapValues(list)

    def reduce_func(key, values):
        values = sorted(values, key=lambda x: x[0])  # sort the values by doc_id
        for i in range(len(values)):  # for each document
            doc_id1, doc1 = values[i]  # get the document id and the document
            doc1 = np.array(doc1)  # convert the document to a numpy array
            for j in range(i + 1, len(values)):  # for each other document
                doc_id2, doc2 = values[j]  # get the document id and the document
                doc2 = np.array(doc2)  # convert the document to a numpy array
                # if the key is not the maximum of the intersection of doc1 and doc2 skip this iteration, this
                # guarantees that two documents are only compared once
                if key != max(np.intersect1d(doc1, doc2)):
                    continue
                # count the number of shared pieces between doc1 and doc2
                shared_pieces = dm.count_shared_pieces(doc1, doc2)
                # if the number of shared pieces is greater than or equal to half of the length of doc1
                if shared_pieces >= len(doc1) // 2:
                    # compute the hamming distance between doc1 and doc2
                    hamming = dm.compute_hamming_distance(doc1, doc2)
                    # if the hamming distance is less than or equal to s bits
                    if hamming <= s:
                        # compute the cosine similarity between doc1 and doc2
                        similarity = dm.compute_cosine_similarity(hamming, m)
                        yield (doc_id1, doc_id2), similarity

    return grouped.flatMap(lambda x: reduce_func(x[0], x[1]))  # return the reduced RDD


def spark_main(path="/home/claudio.lucchese/datasets/enron/*", m=64, p=8, s=3):
    spark = SparkSession.builder.appName('SimHash').getOrCreate()  # create a Spark session
    # document generation for testing
    docs = dm.generate_synthetic_doc_list(spark).toDF(['index', 'text'])
    print(docs.take(10))

    # document generation on real data
    #docs = dm.generate_doc_list(spark, path)  # generate a list of documents
    #docs.show(10)

    comp_time = time.perf_counter()
    mapped, simhash_pieces = mapper(spark, docs, m, p)  # map phase
    comp_time = time.perf_counter() - comp_time
    print('\nmap phase time', comp_time, '\n')
    print(mapped.take(10))
    comp_time = time.perf_counter()
    reduced = reducer(mapped, simhash_pieces, m, s)  # reduce phase
    comp_time = time.perf_counter() - comp_time
    print('\nreduce phase time', comp_time, '\n')
    print(reduced.take(10))
    print('\nnumber of similar pairs', reduced.count())
    spark.stop()


if __name__ == "__main__":
    #load_dotenv()
    #os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
    #pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
    #if pyspark_python:
    #    print(f"PySpark is using this Python interpreter: {pyspark_python}")
    #else:
    #    print("PySpark is using the system's default Python interpreter.")

    if len(sys.argv) == 1:  # if no arguments are provided
        spark_main()  # call the main function with default arguments
    elif len(sys.argv) == 2:  # if 1 argument is provided
        # call the main function with onlt the path argument
        spark_main(sys.argv[1])
    elif len(sys.argv) == 5:  # if 4 arguments are provided
        # call the main function with all the arguments
        spark_main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    else:  # if an invalid number of arguments is provided
        print("Usage: spark-submit main.py <path> [<m> <p> <s>]")
        sys.exit(1)
