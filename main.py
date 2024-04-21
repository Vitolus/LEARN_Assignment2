import doc_manipulation as dm
import time

import gc


if __name__ == "__main__":
    docs = dm.generate_doc_list()
    # docs = dm.generate_synthetic_doc_list()
    # print(docs)
    dt_matrix, terms = dm.compute_doc_term_matrix(docs)
    del docs
    gc.collect()
    m = 128
    dist = set()
    # get number of distinct documents
    for i, j in zip(*dt_matrix.nonzero()):
        dist.add(i)
    n_docs = len(dist)
    del dist
    gc.collect()
    simhash_time = time.perf_counter()
    simhash = dm.compute_simhash(dt_matrix, n_docs, terms.shape[0], m)
    simhash_time = time.perf_counter() - simhash_time
    print(simhash_time)
    p = m/32
    simhash_pieces = dm.split_simhash(simhash, p)
    simhash_ints = dm.pieces_to_ints(simhash_pieces)
    hamming_distances = dm.compute_hamming_distances(simhash_ints)
    cosine_similarities = dm.compute_cosine_similarities(hamming_distances, m)
