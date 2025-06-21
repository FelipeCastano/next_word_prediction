import numpy as np 
import string 
import pandas as pd
from conllu import parse_incr
from pathlib import Path
# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

espanish_noun_suffix = ["ción", "sión", "dad", "tud", "ez", "eza", "aje", "al", "or", "ista", "ero", "ía", "ismo", "ezno", "umbre"]
espanish_verb_suffix = ["ar", "er", "ir", "ecer", "izar", "ificar"]
espanish_adj_suffix = ["able", "ible", "oso", "al", "ico", "il", "ar", "ente", "ivo", "ario", "az", "ón"]
espanish_adv_suffix = ["mente"]

upos_to_ptb = {
    "NOUN": "NN",      # Common noun, singular or mass
    "PROPN": "NNP",    # Proper noun, singular
    "VERB": "VB",      # Base form of verb (e.g., "run")
    "AUX": "VB",       # Auxiliary verb (e.g., "have", "be") simplified to base form
    "ADJ": "JJ",       # Adjective (e.g., "blue")
    "ADV": "RB",       # Adverb (e.g., "quickly")
    "PRON": "PRP",     # Personal pronoun (e.g., "he", "they")
    "DET": "DT",       # Determiner (e.g., "the", "some")
    "ADP": "IN",       # Preposition or subordinating conjunction (e.g., "in", "of", "that")
    "CCONJ": "CC",     # Coordinating conjunction (e.g., "and", "but")
    "SCONJ": "IN",     # Subordinating conjunction, mapped to "IN" (same as ADP)
    "NUM": "CD",       # Cardinal number (e.g., "one", "42")
    "PART": "RP",      # Particle (e.g., "up" in "give up")
    "INTJ": "UH",      # Interjection (e.g., "uh", "wow")
    "PUNCT": ".",      # Punctuation (simplified to comma — adjust as needed)
    "SYM": "SYM",      # Symbol (e.g., "$", "%")
    "X": "FW",          # Foreign word or other uncategorized token
    "\n": ""          # Foreign word or other uncategorized token
}

def assign_unk_espanish(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in espanish_noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in espanish_verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in espanish_adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in espanish_adv_suffix):
        return "--unk_adv--"
    return "--unk--"

def assign_unk_english(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"

def get_word_tag(line, vocab, english): 
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab: 
            # Handle unknown words
            if(english):
                word = assign_unk_english(word)
            else:
                word = assign_unk_espanish(word)
        return word, tag
    return None 

def load_data(path, is_corpus):
    '''
    Loads data from a given file.

    Input:
        path: path to the file
        is_corpus: boolean flag to determine if the file is a corpus (for vocab creation)
    Output:
        If is_corpus is True: returns a vocabulary dictionary mapping word -> index
        If is_corpus is False: returns the file lines as a list
    '''
    with open(path, 'r') as f:
        if is_corpus:
            vocab = {}
            file = f.read().split('\n')
            for i, word in enumerate(sorted(file)):
                vocab[word] = i
            return vocab
        else:
            file = [line for line in f.readlines()]
            return file
    
def conllu_to_dataframe(filepath):
    """
    Parses a CoNLL-U file and returns a pandas DataFrame with columns:
    'sentence_id', 'word', 'lemma', 'upos', 'xpos', 'feats'
    """
    data = []
    sentence_id = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            for token in tokenlist:
                if isinstance(token['id'], int):  # skip multi-word tokens
                    data.append({
                        "sentence_id": sentence_id,
                        "word": token["form"],
                        "lemma": token["lemma"],
                        "upos": token["upos"],  # Universal POS
                        "xpos": token["xpos"],  # Language-specific POS
                        "feats": token["feats"]  # Morphological features
                    })
            data.append({
                "sentence_id": sentence_id,
                "word": '\n',
                "lemma": '\n',
                "upos": '\n',  # Universal POS
                "xpos": '\n',  # Language-specific POS
                "feats": '\n'  # Morphological features
            })
            sentence_id += 1
    return pd.DataFrame(data)

def load_data_english(train_path, test_path, corpus_path):
    training_corpus = load_data(train_path, False)
    y = load_data(test_path, False)
    vocab = load_data(corpus_path, True)
    return training_corpus, y, vocab
    
def load_conllu_data(train_file, test_file):
    # Loading training and test data
    df_train = conllu_to_dataframe(train_file)
    df_train['word'] = df_train['word'].apply(lambda x: x if len(x)<2 else x.replace('"','').replace("'", '').replace('(', '').replace(')', ''))
    df_test = conllu_to_dataframe(test_file)
    df_test['word'] = df_test['word'].apply(lambda x: x if len(x)<2 else x.replace('"','').replace("'", '').replace('(', '').replace(')', ''))

    # Getting upos to ptb translation
    df_train = df_train[['word', 'upos']]
    df_train['label'] = df_train['upos'].apply(lambda x: upos_to_ptb[x])
    df_test = df_test[['word', 'upos']]
    df_test['label'] = df_test['upos'].apply(lambda x: upos_to_ptb[x])

    # Building training corpus
    training_corpus = (df_train['word']+'\t'+df_train['label']+'\n').tolist()
    y = (df_test['word']+'\t'+df_test['label']+'\n').tolist()

    # Building vocab
    unk_list = ["--unk_digit--", "--unk_punct--", "--unk_upper--", "--unk_noun--", "--unk_verb--", "--unk_adj--", "--unk_adv--", "--unk--"]
    vocab_list = df_train['word'].unique().tolist()
    vocab_list.extend(unk_list)
    vocab = {}
    for i, word in enumerate(sorted(vocab_list)):
        vocab[word] = i
    return training_corpus, y, vocab