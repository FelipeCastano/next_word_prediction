from fastapi import FastAPI
from pydantic import BaseModel
from model.ngrams import Ngrams
from model.utils import (
    load_data, 
    tokenize_sentences, 
    count_words, 
    get_words_with_nplus_frequency, 
    replace_oov_words_by_unk
)

def init_ngrams_en():
    """
    Initialize an N-grams model for English language.

    Loads the training data, tokenizes sentences, builds a vocabulary filtering
    out infrequent words, replaces out-of-vocabulary words with '<unk>',
    and finally creates the Ngrams model.

    Returns:
        Ngrams: A trained Ngrams model for English.
    """
    train_path = "data/english_tweets_train.csv"
    train_data = load_data(train_path, 800000)
    train_tokenized = tokenize_sentences(train_data, True)
    vocabulary = get_words_with_nplus_frequency(train_tokenized, count_threshold=1)
    train_data_replaced = replace_oov_words_by_unk(train_tokenized, vocabulary, unknown_token='<unk>')
    k = 1
    ngram = Ngrams(k)
    ngram.create_n_grams(train_data_replaced, vocabulary)
    return ngram

def init_ngrams_es():
    """
    Initialize an N-grams model for Spanish language.

    Loads the training data, tokenizes sentences, builds a vocabulary filtering
    out infrequent words, replaces out-of-vocabulary words with '<unk>',
    and finally creates the Ngrams model.

    Returns:
        Ngrams: A trained Ngrams model for Spanish.
    """
    train_path = "data/spanish_tweets_train.csv"
    train_data = load_data(train_path, 800000)
    train_tokenized = tokenize_sentences(train_data, False)
    vocabulary = get_words_with_nplus_frequency(train_tokenized, count_threshold=1)
    train_data_replaced = replace_oov_words_by_unk(train_tokenized, vocabulary, unknown_token='<unk>')
    k = 1
    ngram = Ngrams(k)
    ngram.create_n_grams(train_data_replaced, vocabulary)
    return ngram

def preprocess_text(text, lang, ngram):
    """
    Preprocesses text for the N-grams model: tokenizes and replaces OOV words.

    Args:
        text (str): Input text.
        lang (bool): True for English, False for Spanish.
        ngram (Ngrams): Corresponding Ngrams model.

    Returns:
        List[str]: List of preprocessed tokens.
    """
    text = tokenize_sentences([text], lang)
    text = replace_oov_words_by_unk(text, ngram.vocabulary, unknown_token='<unk>')
    return text[0]

# Initialize FastAPI app and N-grams models
app = FastAPI()
ngram_en = init_ngrams_en()
ngram_es = init_ngrams_es()

class TagRequest(BaseModel):
    """
    Request model to get next word suggestions.

    Attributes:
        text (str): Input text.
        lang (int): Language of the input text (1 = English, 2 = Spanish).
        starts_with (str): Optional prefix the suggested word should start with.
    """
    text: str  
    lang: int  # 1 = English, 2 = Spanish
    starts_with: str

@app.post("/get_next_word")
def get_next_word(request: TagRequest):
    """
    Endpoint to suggest the top three most probable next words for a given input text.

    Args:
        request (TagRequest): Request object containing text, language, and optional prefix.

    Returns:
        dict: Dictionary with the top three suggested words and their probabilities.
    """
    ngram = ngram_en if request.lang == 1 else ngram_es
    lang = request.lang == 1
    prep = preprocess_text(request.text, lang, ngram)
    r = ngram.top_3_suggestions(prep, request.starts_with)
    result = {
        '1': {"word": r[0][0], "prob": r[0][1]},
        '2': {"word": r[1][0], "prob": r[1][1]},
        '3': {"word": r[2][0], "prob": r[2][1]},
    }
    return result
