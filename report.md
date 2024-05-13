# Assignment 2 - SimHash

### 1. Algorithm

To solve the task of finding near-duplicate documents, we implemented a MapReduce algorithm in pyspark using the
functional programming paradigm. The process compute SimHash to approximate the similarity between documents, and it is
divided into two phases.

The first is the map phase: it receives a collection of documents, and it transforms it into a matrix of documents times
terms, where each cell contains the tf-idf value of the term in the document. Then, to compute the signature of each 
document, we randomly generate a set of vectors of -1 and 1, and for each term in the document, we compute the dot 
product between the term vector and the document vector. If the result is greater than 0, we set the corresponding bit 
in the signature to 1, otherwise, we set it to 0. Moreover, to reduce the dimensionality of the signature, we divide it 
into pieces and convert each piece into an integer. Finally, we apply a map function to the collection of signatures to
emit each docID with each of its pieces of the signature.

The second is the reduce phase: it receives the output of the map phase, it joins each pair  (docID, piece) with the
entire signature of the document, then it groups by piece to get all the documents that have a piece in common. Moreover,
it applies a map function that for each pair of documents, it counts the number of pieces that are equal, and it computes 
the hamming distance between the two signatures only if the number of shared pieces is greater than or equal to 1 but 
less than the total number of pieces. In the latter case, the two documents are equal, so we count them as duplicates. 
Finally, we compute the actual similarity between the two signature, and we emit the pair of documents along the 
similarity value.

The space and time complexity of the MapReduce depend on several factors, including the number of documents, the 
number of unique pieces, and the distribution of pieces among documents.

#### Space Complexity:

The mapper function’s space complexity is determined by the size of the simhash_pieces RDD, which is 
proportional to the number of documents. Therefore, the space complexity is O(n), where n is the number of documents.

The reducer function’s space complexity is determined by the size of the grouped RDD. In the worst case, every
document shares a piece with every other document, so the space complexity is O(n^2). However, in practice, it’s likely 
to be much less than this because not all documents will share a piece.

#### Time Complexity:

The mapper function’s time complexity is determined by the computation of the SimHash and the splitting of the
SimHash into pieces. Both of these operations are linear with respect to the number of documents and the number of terms,
so the time complexity is O(n*m), where n is the number of documents and m is the number of terms.

The reducer function’s time complexity is determined by the grouping operation and the nested loop that 
computes the cosine similarity for each pair of documents. The grouping operation is linear with respect to the number 
of documents, and the nested loop is quadratic with respect to the number of documents that share a piece. Therefore, 
the time complexity is O(n + k^2), where n is the number of documents and k is the maximum number of documents that 
share a piece.

### 2. Testing

All testings are performed on the university's cluster using the provided dataset of emails. The performance and scalability of 
the algorithm is evaluated by measuring the speedup of the version with more workers with respect to the version with
one worker. The execution time is measured with perf_counter() function of the time library.

