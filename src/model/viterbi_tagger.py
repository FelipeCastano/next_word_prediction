import json
import string
import numpy as np
import math
from collections import defaultdict
from model.utils import get_word_tag

class ViterbiTagger:
    '''
    Encapsulates the Viterbi algorithm and helper utilities for POS tagging.
    Existing docstrings have been preserved verbatim. Any new comments are in English.
    '''

    def __init__(self, vocab, alpha: float = 0.001):
        '''
        Initializes the tagger with a smoothing factor.

        Input:
            alpha: smoothing factor for add‑alpha smoothing in probability matrices
        '''
        self.vocab = vocab
        self.alpha = alpha
        self.emission_counts = None
        self.transition_counts = None 
        self.tag_counts = None
        self.states = None
        self.A = None
        self.best_probs = None
        self.best_paths = None

    def create_dictionaries(self, training_corpus, vocab, verbose=True):
        """
        Input: 
            training_corpus: a corpus where each line has a word followed by its tag.
            vocab: a dictionary where keys are words in vocabulary and value is an index
        Output: 
            emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
            transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
            tag_counts: a dictionary where the keys are the tags and the values are the counts
        """
        emission_counts = defaultdict(int)
        transition_counts = defaultdict(int)
        tag_counts = defaultdict(int)
        prev_tag = '--s--' 
        i = 0 
        for word_tag in training_corpus:
            i += 1
            if i % 50000 == 0 and verbose:
                print(f"word count = {i}")
            word, tag = get_word_tag(word_tag, vocab, True)
            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, word)] += 1
            tag_counts[tag] += 1
            prev_tag = tag
        self.emission_counts = emission_counts
        self.transition_counts = transition_counts 
        self.tag_counts = tag_counts
        self.states = sorted(tag_counts.keys())
    
    def predict_pos(self, prep, y):
        '''
        Input: 
            prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
            y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
            emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
            vocab: a dictionary where keys are words in vocabulary and value is an index
            states: a sorted list of all possible tags for this assignment
        Output: 
            accuracy: Number of times you classified a word correctly
        '''
        num_correct = 0
        all_words = set(self.emission_counts.keys())
        total = 0
        for word, y_tup in zip(prep, y): 
            y_tup_l = y_tup.split()
            if len(y_tup_l) == 2:
                true_label = y_tup_l[1]
            else:
                continue
            count_final = 0
            pos_final = ''
            if word in self.vocab:
                for pos in self.states:
                    key = (pos, word)
                    if key in all_words:
                        count = self.emission_counts[key]
                        if count > count_final:
                            count_final = count
                            pos_final = pos
                if pos_final == true_label:
                    num_correct += 1
            total += 1
        accuracy = num_correct / total
        return accuracy

    def create_transition_matrix(self):
        ''' 
        Input: 
            alpha: number used for smoothing
            tag_counts: a dictionary mapping each tag to its respective count
            transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        Output:
            A: matrix of dimension (num_tags,num_tags)
        '''
        all_tags = sorted(self.tag_counts.keys())
        num_tags = len(all_tags)
        A = np.zeros((num_tags,num_tags))
        trans_keys = set(self.transition_counts.keys())
        for i in range(num_tags):
            for j in range(num_tags):
                count = 0
                key = (all_tags[i], all_tags[j])
                if key in trans_keys:
                    count = self.transition_counts[key]
                count_prev_tag = self.tag_counts[key[0]]
                A[i,j] = (count + self.alpha) / ( count_prev_tag + (self.alpha * num_tags))
        self.A = A


    def create_emission_matrix(self):
        '''
        Input: 
            alpha: tuning parameter used in smoothing 
            tag_counts: a dictionary mapping each tag to its respective count
            emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
            vocab: a dictionary where keys are words in vocabulary and value is an index.
                   within the function it'll be treated as a list
        Output:
            B: a matrix of dimension (num_tags, len(vocab))
        '''
        vocab_list = list(self.vocab)
        num_tags = len(self.tag_counts)
        all_tags = sorted(self.tag_counts.keys())
        num_words = len(self.vocab)
        B = np.zeros((num_tags, num_words))
        emis_keys = set(list(self.emission_counts.keys()))
        for i in range(num_tags):
            for j in range(num_words):
                count = 0 
                key = (all_tags[i], vocab_list[j]) # tuple of form (tag,word)
                if key in emis_keys:
                    count = self.emission_counts[key]
                count_tag = self.tag_counts[key[0]]
                B[i,j] = (count + self.alpha) / (count_tag + (num_words * self.alpha))
        self.B = B

    def initialize(self, corpus):
        '''
        Input: 
            states: a list of all possible parts-of-speech
            tag_counts: a dictionary mapping each tag to its respective count
            A: Transition Matrix of dimension (num_tags, num_tags)
            B: Emission Matrix of dimension (num_tags, len(vocab))
            corpus: a sequence of words whose POS is to be identified in a list 
            vocab: a dictionary where keys are words in vocabulary and value is an index
        Output:
            best_probs: matrix of dimension (num_tags, len(corpus)) of floats
            best_paths: matrix of dimension (num_tags, len(corpus)) of integers
        '''
        num_tags = len(self.tag_counts)
        self.best_probs = np.zeros((num_tags, len(corpus)))
        self.best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
        s_idx = self.states.index("--s--")
        for i in range(num_tags):
            self.best_probs[i,0] = math.log(self.A[s_idx, i]) + math.log(self.B[i, self.vocab[corpus[0]]])


    def viterbi_forward(self, test_corpus, verbose=True):
        '''
        Input: 
            A, B: The transition and emission matrices respectively
            test_corpus: a list containing a preprocessed corpus
            best_probs: an initilized matrix of dimension (num_tags, len(corpus))
            best_paths: an initilized matrix of dimension (num_tags, len(corpus))
            vocab: a dictionary where keys are words in vocabulary and value is an index 
        Output: 
            best_probs: a completed matrix of dimension (num_tags, len(corpus))
            best_paths: a completed matrix of dimension (num_tags, len(corpus))
        '''
        num_tags = self.best_probs.shape[0]
        for i in range(1, len(test_corpus)): 
            if i % 5000 == 0 and verbose:
                print("Words processed: {:>8}".format(i))
            for j in range(num_tags):
                best_prob_i = float("-inf")
                best_path_i = None
                for k in range(num_tags):
                    prob = self.best_probs[k, i-1] + math.log(self.A[k,j]) + math.log(self.B[j,self.vocab[test_corpus[i]]])
                    if prob > best_prob_i:
                        best_prob_i = prob
                        best_path_i = k
                self.best_probs[j,i] = best_prob_i
                self.best_paths[j,i] = best_path_i

    def viterbi_backward(self, corpus):
        '''
        This function returns the best path.
        
        '''
        m = self.best_paths.shape[1] 
        z = [None] * m # DO NOT replace the "None"
        num_tags = self.best_probs.shape[0]
        best_prob_for_last_word = float('-inf')
        pred = [None] * m
        for k in range(num_tags):
            if self.best_probs[k, -1] > best_prob_for_last_word: 
                best_prob_for_last_word = self.best_probs[k, -1]
                z[m - 1] = k
        pred[m - 1] = self.states[z[m - 1]]
        for i in range(m-1, 0, -1):
            pos_tag_for_word_i = z[i]
            z[i - 1] = self.best_paths[pos_tag_for_word_i, i]
            pred[i - 1] = self.states[z[i - 1]]
        return pred
    
    @staticmethod
    def compute_accuracy(pred, y):
        '''
        Computes the accuracy of predictions against ground truth.

        Input:
            pred: list of predicted POS tags
            y: list of labeled lines ("word tag")
        Output:
            accuracy: float
        '''
        num_correct = 0
        total = 0

        for prediction, y in zip(pred, y):
            word_tag_tuple = y.split()
            if len(word_tag_tuple) != 2:
                continue
            word, tag = word_tag_tuple[0], word_tag_tuple[1]
            if prediction == tag:
                num_correct += 1
            total += 1

        return num_correct / total

    def build_matrices(self, training_corpus, vocab):
        '''
        Builds and stores the transition (A) and emission (B) matrices using the tagger's alpha value.
    
        Input:
            tag_counts: dict mapping each tag to its count
            transition_counts: dict with (prev_tag, tag) tuples as keys and their counts as values
            emission_counts: dict with (tag, word) tuples as keys and their counts as values
            vocab: dict mapping word -> index
    
        Output:
            None (matrices are stored internally in self.A and self.B)
        '''
        self.create_dictionaries(training_corpus, vocab)
        self.A = self.create_transition_matrix(self.alpha, self.tag_counts, self.transition_counts)
        self.B = self.create_emission_matrix(self.alpha, self.tag_counts, self.emission_counts, vocab)
        self.vocab = vocab
        self.states = sorted(tag_counts.keys())


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
