import doc_manipulation as dm
import time
import gc


if __name__ == "__main__":
    docs = dm.generate_doc_list()
    #docs = dm.generate_synthetic_doc_list()
    print(docs)
    dt_matrix, terms = dm.compute_doc_term_matrix(docs)
    del docs
    gc.collect()
    m = 128
    comp_time = time.perf_counter()
    simhash = dm.compute_simhash(dt_matrix, terms.shape[0], m)
    comp_time = time.perf_counter() - comp_time
    print('\nsimhash time', comp_time, '\n')
    p = m/32
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
