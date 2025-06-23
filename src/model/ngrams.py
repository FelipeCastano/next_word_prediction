from itertools import chain
import numpy as np
import pandas as pd

class Ngrams:
    def __init__(self):
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.vocabulary = []
        self.vocabulary_size = 0
        self.count_matrix = []
        self.prob_matrix = []
        self.start_token='<s>'
        self.end_token = '<e>'
        self.unknown_token = '<unk>'

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
                n_gram = sentence[i:i+n]
                if n_gram in n_grams:
                    n_grams[n_gram] += 1
                else:
                    n_grams[n_gram] = 1
        return n_grams

    def create_n_grams(self, sentences):
        self.unigram_counts = self.count_n_grams(sentences, 1)
        self.bigram_counts = self.count_n_grams(sentences, 2)
        self.vocabulary = list(set(chain.from_iterable(sentences)))
        self.vocabulary = self.vocabulary + [self.end_token, self.unknown_token]    
        self.vocabulary_size = len(self.vocabulary)

    def estimate_probability(self, word, previous_n_gram, k=1.0):
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
        denominator = previous_n_gram_count+k*self.vocabulary_size
        n_plus1_gram = previous_n_gram + (word,) 
        n_plus1_gram_count = self.bigram_counts.get(n_plus1_gram, 0)
        numerator = n_plus1_gram_count+k
        probability = numerator/denominator
        return probability
    
    def estimate_probabilities(self, previous_n_gram, k=1.0):
        """
        Estimate the probabilities of next words using the n-gram counts with k-smoothing
        
        Args:
            previous_n_gram: A sequence of words of length n
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary: List of words
            k: positive constant, smoothing parameter
        
        Returns:
            A dictionary mapping from next words to the probability.
        """
        previous_n_gram = tuple(previous_n_gram)
        probabilities = {}
        for word in self.vocabulary:
            probability = self.estimate_probability(word, previous_n_gram, k=k)
            probabilities[word] = probability
        return probabilities

    def make_count_matrix(self):
        n_grams = []
        for n_plus1_gram in self.bigram_counts.keys():
            n_gram = n_plus1_gram[0:-1]        
            n_grams.append(n_gram)
        n_grams = list(set(n_grams))
        
        row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}    
        col_index = {word:j for j, word in enumerate(self.vocabulary)}    
        
        nrow = len(n_grams)
        ncol = len(self.vocabulary)
        count_matrix = np.zeros((nrow, ncol))
        for n_plus1_gram, count in self.bigram_counts.items():
            n_gram = n_plus1_gram[0:-1]
            word = n_plus1_gram[-1]
            if word not in self.vocabulary:
                continue
            i = row_index[n_gram]
            j = col_index[word]
            count_matrix[i, j] = count
        count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=self.vocabulary)
        self.count_matrix = count_matrix

    def make_probability_matrix(self, k=1.0):
        count_matrix = self.count_matrix
        count_matrix += k
        prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
        self.prob_matrix = prob_matrix