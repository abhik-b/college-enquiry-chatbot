import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
from flask_server.university.models import Course

stemmer=PorterStemmer()
 
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


import spacy 
nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher


matcher=Matcher(nlp.vocab)

courses=Course.query.all()

for course in courses:
    pattern = [{"LOWER":course.name.lower()}]
    matcher.add(course.name,[pattern])

btech_pattern = [
    [{"LOWER":"b."},{"LOWER":"tech"}],
    [{"LOWER":"b"},{"LOWER":"tech"}],
    [{"LOWER":"btech"}]
]

matcher.add("mtech",[[{"LOWER":"mtech"}],[{"LOWER":"m","OP":"+"},{"LOWER":"tech"}]])
matcher.add("btech",btech_pattern)

def course_matcher(sentence):
    doc = nlp(sentence)
    matches = matcher(doc)

    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # 'mtech or btech'
        return string_id




