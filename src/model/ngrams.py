from itertools import chain
import numpy as np
import pandas as pd

class Ngrams:
    def __init__(self, k):
        """
        Initializes the N-grams model with Laplace smoothing.

        Args:
            k (float): Smoothing parameter (k > 0).
        """
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.vocabulary = []
        self.vocabulary_size = 0
        self.count_matrix = []
        self.prob_matrix = []
        self.start_token = '<s>'
        self.end_token = '<e>'
        self.unknown_token = '<unk>'
        self.k = k

    def count_n_grams(self, data, n):
        """
        Counts all n-grams in the data.

        Args:
            data (List[List[str]]): List of sentences, each sentence is a list of words.
            n (int): Length of the n-gram.

        Returns:
            dict: A dictionary mapping n-gram tuples to their frequency counts.
        """
        n_grams = {}
        for sentence in data:
            sentence = [self.start_token] * n + sentence + [self.end_token]
            sentence = tuple(sentence)
            for i in range(len(sentence) - n + 1):
                n_gram = sentence[i:i + n]
                n_grams[n_gram] = n_grams.get(n_gram, 0) + 1
        return n_grams

    def create_n_grams(self, sentences, vocabulary):
        """
        Creates unigram and bigram counts from the sentences.

        Args:
            sentences (List[List[str]]): List of sentences, each sentence is a list of words.
            vocabulary (List[str]): List of words forming the vocabulary.
        """
        self.unigram_counts = self.count_n_grams(sentences, 1)
        self.bigram_counts = self.count_n_grams(sentences, 2)
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)

    def estimate_probability(self, word, previous_n_gram):
        """
        Estimates the probability of the next word using Laplace-smoothed n-gram counts.

        Args:
            word (str): The next word to estimate.
            previous_n_gram (List[str] | Tuple[str]): The preceding n-gram sequence.

        Returns:
            float: The smoothed probability that `word` follows `previous_n_gram`.
        """
        previous_n_gram = tuple(previous_n_gram)
        previous_n_gram_count = self.unigram_counts.get(previous_n_gram, 0)
        denominator = previous_n_gram_count + self.k * self.vocabulary_size
        n_plus1_gram = previous_n_gram + (word,)
        n_plus1_gram_count = self.bigram_counts.get(n_plus1_gram, 0)
        numerator = n_plus1_gram_count + self.k
        probability = numerator / denominator
        return probability

    def estimate_probabilities(self, previous_n_gram):
        """
        Estimates probabilities for all possible next words given a previous n-gram.

        Args:
            previous_n_gram (List[str] | Tuple[str]): The preceding n-gram sequence.

        Returns:
            dict: Dictionary mapping each vocabulary word to its conditional probability.
        """
        previous_n_gram = tuple(previous_n_gram)
        probabilities = {}
        for word in self.vocabulary:
            probabilities[word] = self.estimate_probability(word, previous_n_gram)
        return probabilities

    def make_count_matrix(self):
        """
        Creates a count matrix (for bigrams) where rows represent n-grams and columns represent next words.
        The resulting matrix is stored in `self.count_matrix`.
        """
        n_grams = []
        for n_plus1_gram in self.bigram_counts.keys():
            n_gram = n_plus1_gram[:-1]
            n_grams.append(n_gram)
        n_grams = list(set(n_grams))

        row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
        col_index = {word: j for j, word in enumerate(self.vocabulary)}

        nrow = len(n_grams)
        ncol = len(self.vocabulary)
        count_matrix = np.zeros((nrow, ncol))

        for n_plus1_gram, count in self.bigram_counts.items():
            n_gram = n_plus1_gram[:-1]
            word = n_plus1_gram[-1]
            if word not in self.vocabulary:
                continue
            i = row_index[n_gram]
            j = col_index[word]
            count_matrix[i, j] = count

        self.count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=self.vocabulary)

    def make_probability_matrix(self):
        """
        Creates a probability matrix from the count matrix using Laplace smoothing.
        The result is stored in `self.prob_matrix`.
        """
        count_matrix = self.count_matrix.copy()
        count_matrix += self.k
        prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
        self.prob_matrix = prob_matrix

    def calculate_perplexity(self, sentence):
        """
        Calculates the perplexity of a given sentence under the current model.

        Args:
            sentence (List[str]): List of words forming the sentence.

        Returns:
            float: Perplexity value of the sentence.
        """
        n = len(list(self.unigram_counts.keys())[0])
        sentence = [self.start_token] * n + sentence + [self.end_token]
        sentence = tuple(sentence)
        N = len(sentence)
        product_pi = 1.0
        for t in range(n, N):
            n_gram = sentence[t - n:t]
            word = sentence[t]
            probability = self.estimate_probability(word, n_gram)
            product_pi *= 1 / probability
        perplexity = product_pi ** (1 / N)
        return perplexity

    def suggest_a_word(self, previous_tokens, start_with=None):
        """
        Suggests the most probable next word following an input sequence.

        Args:
            previous_tokens (List[str]): List of previous tokens (words), must have at least length n.
            start_with (str, optional): If specified, filters suggestions to words starting with this prefix.

        Returns:
            Tuple[str, float]: Tuple with the suggested word and its probability.
        """
        n = len(list(self.unigram_counts.keys())[0])
        previous_tokens = [self.start_token] * n + previous_tokens
        previous_n_gram = previous_tokens[-n:]
        probabilities = self.estimate_probabilities(previous_n_gram)

        suggestion = None
        max_prob = 0
        for word, prob in probabilities.items():
            if start_with is not None and not word.startswith(start_with):
                continue
            if prob >= max_prob:
                suggestion = word
                max_prob = prob
        return suggestion, max_prob

    def top_3_suggestions(self, previous_tokens, start_with=None):
        """
        Returns the top three most probable next words following an input sequence.

        Args:
            previous_tokens (List[str]): List of previous tokens (words), must have at least length n.
            start_with (str, optional): If specified, filters suggestions to words starting with this prefix.

        Returns:
            List[Tuple[str, float]]: List of up to three tuples (word, probability), ordered by descending probability.
        """
        n = len(list(self.unigram_counts.keys())[0])
        previous_tokens = [self.start_token] * n + previous_tokens
        previous_n_gram = previous_tokens[-n:]
        probabilities = self.estimate_probabilities(previous_n_gram)
        top3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        return top3
