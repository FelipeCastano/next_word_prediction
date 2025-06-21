from fastapi import FastAPI
from pydantic import BaseModel
from model.viterbi_tagger import ViterbiTagger
from model.utils import assign_unk_english, assign_unk_espanish, load_conllu_data, load_data_english
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


def init_viterbi_tagger_en():
    train_path = "data/WSJ_02-21.pos"
    test_path = "data/WSJ_24.pos"
    corpus_path = "data/hmm_vocab.txt"
    training_corpus, y, vocab = load_data_english(train_path, test_path, corpus_path)
    alpha = 0.001
    tagger = ViterbiTagger(vocab, alpha)
    tagger.create_dictionaries(training_corpus, vocab)
    tagger.create_transition_matrix()
    tagger.create_emission_matrix()
    return tagger

def init_viterbi_tagger_es():
    train_file = "data/es_ancora-ud-train.conllu"
    test_file = "data/es_ancora-ud-test.conllu"
    training_corpus, y, vocab = load_conllu_data(train_file, test_file)
    alpha = 0.001
    tagger = ViterbiTagger(vocab, alpha)
    tagger.create_dictionaries(training_corpus, vocab)
    tagger.create_transition_matrix()
    tagger.create_emission_matrix()
    return tagger

def tokenize(text, lang):
    nlp = spacy_en if lang == 1 else spacy_es
    doc = nlp(text)
    return [token.text for token in doc]

def tokenize(text, lang):
    if lang == 1:
        return word_tokenize(text, language='english')
    else:
        return word_tokenize(text, language='spanish')
    
def preprocess_text(text, lang, tagger):
    if lang == 1:
        assign_unk = assign_unk_english 
    else:
        assign_unk = assign_unk_espanish
    
    text = tokenize(text, lang)
    prep = []
    for item in text:
        word = item.split('\t')[0]
        if not word in tagger.vocab.keys():
            prep.append(assign_unk(word))
        else:
            prep.append(word)
    return prep




app = FastAPI()

tagger_en = init_viterbi_tagger_en()
tagger_es = init_viterbi_tagger_es()

class TagRequest(BaseModel):
    text: str
    lang: int  # 1 = English, 2 = Spanish

@app.post("/get_tag")
def get_tag(request: TagRequest):
    tagger = tagger_en if request.lang == 1 else tagger_es
    prep = preprocess_text(request.text, request.lang, tagger)
    tagger.initialize(prep)
    tagger.viterbi_forward(prep)
    pred = tagger.viterbi_backward(prep)
    result = {
      "tokens": prep,
      "tags": pred
    }
    return result

