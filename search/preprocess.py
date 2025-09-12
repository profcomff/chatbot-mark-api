import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem


_MYSTEM = Mystem()
_STEMMER = SnowballStemmer("russian")
_PREPROCESS_REGEX = re.compile(r'[^а-яё\s]')
_STOP_WORDS = set(stopwords.words('russian'))
_BANNED_WORDS = {'мгу', 'физфак', 'физический', 'университет'}
_STEMMED_BANNED_WORDS = {_STEMMER.stem(w) for w in _BANNED_WORDS}
_LEMMATIZED_BANNED_WORDS = {lemma.strip() for w in _BANNED_WORDS for lemma in _MYSTEM.lemmatize(w)}


def preprocess_stem(text, filter_stopwords=True, filter_stemmed_banned_words=True):
    cleaned = _PREPROCESS_REGEX.sub('', text.lower())
    words = word_tokenize(cleaned, language="russian")
    if filter_stopwords:
        words = [w for w in words if w not in _STOP_WORDS]
    stemmed = [_STEMMER.stem(w) for w in words]
    if filter_stemmed_banned_words:
        return [w for w in stemmed if w not in _STEMMED_BANNED_WORDS]
    return stemmed


def preprocess_lemma(text, filter_stopwords=False, filter_lemmatized_banned_words=False):
    cleaned = _PREPROCESS_REGEX.sub('', text.lower())
    words = word_tokenize(cleaned, language="russian")
    if filter_stopwords:
        words = [w for w in words if w not in _STOP_WORDS]
    lemmas = [_MYSTEM.lemmatize(w)[0].strip() for w in words]
    if filter_lemmatized_banned_words:
        return [w for w in lemmas if w not in _LEMMATIZED_BANNED_WORDS]
    return lemmas
