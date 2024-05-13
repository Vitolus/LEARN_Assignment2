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

### 2. Testing


