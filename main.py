import sys
import doc_manipulation as dm
import time
import numpy as np
from pyspark.sql import SparkSession
# from dotenv import load_dotenv
# import os


def mapper(docs, m, p):
    """
    Map Phase:
    - Compute SimHash: For each document, compute the SimHash. This can be done in parallel for all documents.
    - Split SimHash: Split the SimHash of each document into pieces of equal length. This operation is also independent
    for each document and can be done in parallel.
    - Emit Pairs: For each document, emit pairs of the form (piece, document_id). This will allow grouping by piece in
    the next phase.
    """
    global spark
    comp_time = time.perf_counter()
    tfidf, n_terms = dm.compute_tfidf(docs)
    docs.unpersist()
    print('tfidf time', time.perf_counter() - comp_time)
    comp_time = time.perf_counter()
    rw = dm.compute_rw(spark, n_terms, m)
    print('rw time', time.perf_counter() - comp_time)
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(tfidf, rw)
    print('simhash time', time.perf_counter() - comp_time)
    comp_time = time.perf_counter()
    simhash_pieces = dm.split_simhash(simhash, m // p)
    print('split time', time.perf_counter() - comp_time, '\n')

    def map_func(doc):
        doc_id, pieces = doc
        pieces = np.sort(pieces)  # sort the pieces in ascending order
        # take the first half of the pieces to reduce the number of copies across nodes,
        # at the cost of some accuracy in the number of similar pairs
        for piece in pieces[:len(pieces) // 2]:
            yield doc_id, piece

    return simhash_pieces.flatMap(map_func), simhash_pieces  # return the mapped RDD and the simhash pieces


def reducer(mapped, simhash_pieces, m, s):
    """
    Reduce Phase:
    - Group by Piece: Group the documents that have one of the pieces of the SimHash equal.
    - Compute Hamming Distance: For each group of documents (that share at least half of the SimHash pieces), compute
    the Hamming distance between the SimHashes of each pair of documents.
    - Compute Cosine Similarity: Still within each group, compute the cosine similarity for each pair of documents.
    Emit Similar Documents: Finally, for each document, emit the list of documents that are considered similar based on
    the computed cosine similarity and Hamming distance.
    """
    global spark
    joined = mapped.join(simhash_pieces)  # Join on docID
    mapped.unpersist()
    # Group by piece and sort by docID in each group
    grouped = joined.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().mapValues(list)
    n_equals = spark.sparkContext.accumulator(0)  # accumulator to count the number of equal pairs

    def reduce_func(key, values):  # (piece, [(doc_id, doc),...]) pairs
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
                if dm.count_shared_pieces(doc1, doc2) == len(doc1):
                    n_equals.add(1)
                    continue
                similarity = dm.compute_cosine_similarity(dm.compute_hamming_distance(doc1, doc2), m)
                if similarity > s:
                    yield (doc_id1, doc_id2), similarity

    return grouped.flatMap(lambda x: reduce_func(x[0], x[1])), n_equals  # return the reduced RDD


def spark_main(ext="parquet", path="./data/emails/*", n_docs=10000, m=64, p=8, s=0.95):
    global spark
    docs = dm.generate_doc_list(spark, ext, path).limit(n_docs)  # generate a list of documents
    docs.persist()
    print('number of documents:', n_docs)
    print('length of the signature:', m, 'bits')
    print('number of pieces:', p)
    print('similarity threshold:', s * 100, '%\n')
    comp_time = time.perf_counter()
    mapped, simhash_pieces = mapper(docs, m, p)  # map phase
    mapped.persist()
    map_count = mapped.count()
    print('map phase time', time.perf_counter() - comp_time)
    print('number of mapped pairs', map_count, '\n')
    comp_time = time.perf_counter()
    reduced, n_equals = reducer(mapped, simhash_pieces, m, s)  # reduce phase
    reduced.persist()
    reduce_count = reduced.count()
    print('reduce phase time', time.perf_counter() - comp_time)
    print('number of similar pairs', reduce_count)
    print('number of equal pairs', n_equals.value)
    spark.stop()


if __name__ == "__main__":
    # load_dotenv()
    # os.environ["PYSPARK_PYTHON"] = os.getenv("PYSPARK_PYTHON")
    # pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
    # if pyspark_python:
    #     print(f"PySpark is using this Python interpreter: {pyspark_python}")
    # else:
    #     print("PySpark is using the system's default Python interpreter.")
    spark = SparkSession.builder.appName("SimHash").getOrCreate()  # create a Spark session
    spark.sparkContext.setLogLevel('WARN')

    if len(sys.argv) == 1:
        spark_main()
    elif len(sys.argv) == 3:
        spark_main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 7:
        spark_main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
    else:
        print("Usage: [<extension of file> <path>] [<n_docs> <m> <p> <s>]")
        sys.exit(1)
