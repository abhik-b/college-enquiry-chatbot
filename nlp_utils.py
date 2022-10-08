import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np


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



# fetch all course tags from dbms
# then for each of the tags create a pattern
# for those courses where people use different aliases
# we can extend those patterns manually
# Example : 
# print('what is b. tech ?',course_matcher('what is b. tech ?')) ✅
# print('what is btech ?',course_matcher('what is btech ?'))✅
# print('what is b tech ?',course_matcher('what is b tech ?'))✅
# print('what is b.tech ?',course_matcher('what is b.tech ?'))❌

# matcher.add('btech',[
#     [{"LOWER":"b.tech"}]
# ])
# print('what is b.tech ?',course_matcher('what is b.tech ?'))✅