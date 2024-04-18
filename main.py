import doc_manipulation as dm


if __name__ == "__main__":
    docs = dm.generate_doc_list()
    print(docs)
    dt_matrix, terms = dm.compute_doc_term_matrix(docs)
    print(terms)
    print(dt_matrix)
    m = 64
    simhash = dm.compute_simhash(dt_matrix, m)
    p = 2
    simhash_pieces = dm.split_simhash(simhash, p)
    print(simhash_pieces)
    simhash_ints = dm.convert_split_to_int(simhash_pieces)
    print(simhash_ints)
