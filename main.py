import doc_manipulation as dm
import numpy as np


if __name__ == "__main__":
    docs = dm.generate_doc_list()
    dt_matrix, terms = dm.compute_doc_term_matrix(docs)
    m = 64
    simhash = dm.compute_simhash(dt_matrix, m)
    p = 2
    simhash_pieces = dm.split_simhash(simhash, p)
    simhash_ints = dm.pieces_to_ints(simhash_pieces)
    hamming_distances = dm.compute_hamming_distances(simhash_ints)
