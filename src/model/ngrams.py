from itertools import chain
import numpy as np
import pandas as pd

class Ngrams:
    def __init__(self, k):
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.vocabulary = []
        self.vocabulary_size = 0
        self.count_matrix = []
        self.prob_matrix = []
        self.start_token ='<s>'
        self.end_token = '<e>'
        self.unknown_token = '<unk>'
        self.k = k

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

    def create_n_grams(self, sentences, vocabulary):
        self.unigram_counts = self.count_n_grams(sentences, 1)
        self.bigram_counts = self.count_n_grams(sentences, 2)
        self.vocabulary = vocabulary
        #self.vocabulary = self.vocabulary + [self.end_token, self.unknown_token]    
        self.vocabulary_size = len(self.vocabulary)

    def estimate_probability(self, word, previous_n_gram):
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
        denominator = previous_n_gram_count+self.k*self.vocabulary_size
        n_plus1_gram = previous_n_gram + (word,) 
        n_plus1_gram_count = self.bigram_counts.get(n_plus1_gram, 0)
        numerator = n_plus1_gram_count+self.k
        probability = numerator/denominator
        return probability
    
    def estimate_probabilities(self, previous_n_gram):
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
            probability = self.estimate_probability(word, previous_n_gram)
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

    def make_probability_matrix(self):
        count_matrix = self.count_matrix
        count_matrix += self.k
        prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
        self.prob_matrix = prob_matrix

    def calculate_perplexity(self, sentence):
        """
        Calculate perplexity for a list of sentences
        
        Args:
            sentence: List of strings
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary_size: number of unique words in the vocabulary
            k: Positive smoothing constant
        
        Returns:
            Perplexity score
        """
        n = len(list(self.unigram_counts.keys())[0]) 
        sentence = [self.start_token] * n + sentence + [self.end_token]
        sentence = tuple(sentence)
        N = len(sentence)
        product_pi = 1.0
        for t in range(n, N):
            n_gram = sentence[t-n:t]
            word = sentence[t]
            probability = self.estimate_probability(word, n_gram)
            product_pi *= 1/probability
        perplexity = (product_pi)**(1/N)
        return perplexity

    def suggest_a_word(self, previous_tokens, start_with=None):
        """
        Get suggestion for the next word
        
        Args:
            previous_tokens: The sentence you input where each token is a word. Must have length >= n 
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary: List of words
            k: positive constant, smoothing parameter
            start_with: If not None, specifies the first few letters of the next word
            
        Returns:
            A tuple of 
              - string of the most likely next word
              - corresponding probability
        """
        n = len(list(self.unigram_counts.keys())[0])
        previous_tokens = [self.start_token] * n + previous_tokens
        previous_n_gram = previous_tokens[-n:]
        probabilities = self.estimate_probabilities(previous_n_gram)
        suggestion = None
        max_prob = 0
        for word, prob in probabilities.items():
            if start_with is not None: 
                if not word.startswith(start_with): 
                    continue
            if prob >= max_prob: 
                suggestion = word
                max_prob = prob
        return suggestion, max_prob

    def top_3_suggestions(self, previous_tokens, start_with=None):
        """
        Get suggestion for the next word
        
        Args:
            previous_tokens: The sentence you input where each token is a word. Must have length >= n 
            n_gram_counts: Dictionary of counts of n-grams
            n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
            vocabulary: List of words
            k: positive constant, smoothing parameter
            start_with: If not None, specifies the first few letters of the next word
            
        Returns:
            A tuple of 
              - string of the most likely next word
              - corresponding probability
        """
        n = len(list(self.unigram_counts.keys())[0])
        previous_tokens = [self.start_token] * n + previous_tokens
        previous_n_gram = previous_tokens[-n:]
        probabilities = self.estimate_probabilities(previous_n_gram)
        top3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        return top3