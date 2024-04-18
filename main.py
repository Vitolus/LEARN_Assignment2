import doc_manipulation as dm
import numpy as np


if __name__ == "__main__":
    docs = dm.generate_doc_list()
    print(docs)
    dt_matrix, terms = dm.compute_doc_term_matrix(docs)
    print(terms)
    print(dt_matrix)
    m = 64
    # simhash = dm.compute_simhash(dt_matrix, terms, m)
    simhash = dm.compute_simhash()
    print(simhash)