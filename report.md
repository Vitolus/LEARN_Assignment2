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
one worker. The execution time is measured with perf_counter() function of the time library. For all testings has been
used a similarity threshold of 95%; different results on the number fo pairs, at parity of signature and pieces, are due
to the randomness of the algorithm when generating the term vectors.

| Workers | documents | signature | pieces | mapper time | reducer time | total time | similar pairs | duplicates |
|---------|-----------|-----------|--------|-------------|--------------|------------|---------------|------------|
| 4       | 10000     | 64        | 4      | 35.35       | 6.20         | 41.55      | 500           | 1081       |
| 3       | 10000     | 64        | 4      | 32.31       | 6.63         | 38.94      | 415           | 980        |
| 2       | 10000     | 64        | 4      |             |              |
| 1       | 10000     | 64        | 4      |             |              |
| ------- | --------- | --------- | ------ | ----------- | ------------ | ---------- | ------------- | ---------- |
| 4       | 10000     | 64        | 8      | 34.27       | 24.22        | 58.49      | 2067          | 1351       |
| 3       | 10000     | 64        | 8      | 32.74       | 22.93        | 55.67      | 1868          | 1175       |
| 2       | 10000     | 64        | 8      |             |              |
| 1       | 10000     | 64        | 8      |             |              |
| ------- | --------- | --------- | ------ | ----------- | ------------ | ---------- | ------------- | ---------- |
| 4       | 10000     | 64        | 16     | 33.76       | 1599.99      | 1633.75    | 7869          | 9626       |
| 3       | 10000     | 64        | 16     |             |              |
| 2       | 10000     | 64        | 16     |             |              |
| 1       | 10000     | 64        | 16     |             |              |
| ------- | --------- | --------- | ------ | ----------- | ------------ | ---------- | ------------- | ---------- |
| 4       | 10000     | 128       | 4      | 35.95       | 5.90         | 41.85      | 348           | 1126       |
| 4       | 10000     | 128       | 8      | 33.73       | 6.07         | 39.80      | 1128          | 1150       |
| 4       | 10000     | 128       | 16     | 33.49       | 76.32        | 109.81     | 2615          | 1497       |
| 4       | 50000     | 64        | 4      | 114.55      | 20.32        | 134.87     | 9965          | 22692      |
| 4       | 50000     | 64        | 8      | 119.68      | 450.75       | 570.43     | 31364         | 15167      |
| 4       | 50000     | 64        | 16     |
| 4       | 50000     | 128       | 4      | 119.43      | 20.66        | 140.09     | 4917          | 21007      |
| 4       | 50000     | 128       | 8      | 117.68      | 31.82        | 149.50     | 11650         | 22162      |
| 4       | 50000     | 128       | 16     | 117.16      | 1755.31      | 1872.47    | 48654         | 22381      |