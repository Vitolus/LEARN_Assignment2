import doc_manipulation as dm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    docs = dm.generate_doc_list()
    print(docs)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(docs)
    terms = np.array(tfidf.get_feature_names_out())
    print(terms)
    dt_matrix = np.array(tfidf.transform(docs).toarray())
    print(dt_matrix)
    simhash = dm.compute_simhash(dt_matrix, terms)
    print(simhash)