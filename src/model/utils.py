import pandas as pd
import spacy

nlp_en = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
nlp_es = spacy.load("es_core_news_sm", disable=["tagger", "parser", "ner"])

def load_data(path):
    """
    Loads and cleans sentences from a given file.

    Args:
        path (str): Path to the file containing text data.

    Returns:
        list of str: A list of non-empty, stripped sentences from the file. 
        Sentences are expected to be separated by newline characters.

    Raises:
        FileNotFoundError: If the file at the given path does not exist.
        IOError: If an I/O error occurs while reading the file.
    """
    data_df = pd.read_csv(path)
    sentences = data_df['sentence'].tolist()
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences


def tokenize_sentences(sentences, english_mode=True):
    """
    Tokenizes sentences into tokens (words) using spaCy.
    Uses English tokenizer if english_mode is True, Spanish tokenizer otherwise.
    Processes sentences in batch mode for better performance.

    Args:
        sentences (list of str): List of sentences to tokenize.
        english_mode (bool): If True, tokenize in English; if False, tokenize in Spanish.

    Returns:
        list of list of str: Tokenized sentences.
    """
    nlp = nlp_en if english_mode else nlp_es
    sentences = [s.lower() for s in sentences] 
    tokenized_sentences = [
        [token.text for token in doc]
        for doc in nlp.pipe(sentences, batch_size=256, n_process=1)
    ]
    return tokenized_sentences


def count_words(tokenized_sentences):
    """
    Counts the frequency of each word in tokenized sentences.

    Args:
        tokenized_sentences (list of list of str): Tokenized sentences as lists of words.

    Returns:
        dict: A dictionary mapping each word (str) to its frequency (int).
    """
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            if not token in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Finds words that appear at least `count_threshold` times.

    Args:
        tokenized_sentences (list of list of str): Tokenized sentences as lists of words.
        count_threshold (int): Minimum frequency a word must have to be included.

    Returns:
        list of str: Words appearing `count_threshold` times or more.
    """
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)
    for word, cnt in word_counts.items(): 
        if cnt >= count_threshold:
            closed_vocab.append(word)
    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replaces words not in the vocabulary with a specified unknown token.

    Args:
        tokenized_sentences (list of list of str): Tokenized sentences as lists of words.
        vocabulary (list of str): List of known words.
        unknown_token (str, optional): Token to replace out-of-vocabulary words. Defaults to "<unk>".

    Returns:
        list of list of str: Tokenized sentences with out-of-vocabulary words replaced.
    """
    vocabulary = set(vocabulary)
    replaced_tokenized_sentences = []
    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences
