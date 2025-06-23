from itertools import chain

class Ngrams:
    def __init__(self):
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.vocabulary_size = 0
        self.start_token='<s>'
        self.end_token = '<e>'

    def count_n_grams(self, data, n):
        """
        Count all n-grams in the data
        
        Args:
            data: List of lists of words
            n: number of words in a sequence
        
        Returns:
            A dictionary that maps a tuple of n-words to its frequency
        """
        n_grams = {}
        for sentence in data:
            sentence = [self.start_token]*n+sentence+[self.end_token]
            sentence = tuple(sentence)
            for i in range(len(sentence)-n+1):
                self.n_gram = sentence[i:i+n]
                if n_gram in n_grams:
                    n_grams[n_gram] += 1
                else:
                    n_grams[n_gram] = 1
        return n_grams

    def create_n_grams(sentences):
        self.unigram_counts = self.count_n_grams(sentences, 1)
        self.bigram_counts = self.count_n_grams(sentences, 2)
        unique_words = list(set(chain.from_iterable(sentences)))
        self.vocabulary_size = len(unique_words)

    def estimate_probability(word, previous_n_gram, , k=1.0):
        """
        Estimate the probabilities of a next word using the n-gram counts with k-smoothing
        
        Args:
            word: next word
            previous_n_gram: A sequence of words of length n
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary_size: number of words in the vocabulary
            k: positive constant, smoothing parameter
        
        Returns:
            A probability
        """
        previous_n_gram = tuple(previous_n_gram)
        previous_n_gram_count = self.unigram_counts.get(previous_n_gram, 0)
        denominator = previous_n_gram_count+k*vocabulary_size
        n_plus1_gram = previous_n_gram + (word,) 
        n_plus1_gram_count = self.bigram_counts.get(n_plus1_gram, 0)
        numerator = n_plus1_gram_count+k
        probability = numerator/denominator
        return probability
    
    def tag(self, corpus):
        '''
        Performs POS tagging on the given corpus using the internally stored matrices.
        
        Input:
            corpus: list of words (preprocessed)
        
        Output:
            pred: list of predicted POS tags corresponding to the input words
        '''
        if self.A is None or self.B is None:
            raise ValueError("Matrices A and B have not been built. Call build_matrices() first.")
        best_probs, best_paths = self.initialize(
            self.states,
            defaultdict(int, {s: i for i, s in enumerate(self.states)}),
            self.A, self.B, corpus, self.vocab
        )
        best_probs, best_paths = self.viterbi_forward(self.A, self.B, corpus, best_probs, best_paths, self.vocab, verbose=False)
        pred = self.viterbi_backward(best_probs, best_paths, corpus, self.states)
        return pred
