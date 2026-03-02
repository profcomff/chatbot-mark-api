import re

import json
from typing import Dict, Pattern
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from pathlib import Path


_MYSTEM = Mystem()
_STEMMER = SnowballStemmer("russian")
_PREPROCESS_REGEX = re.compile(r'[^а-яё\s]')
_STOP_WORDS = set(stopwords.words('russian'))
_BANNED_WORDS = {'мгу', 'физфак', 'физический', 'университет'}
_STEMMED_BANNED_WORDS = {_STEMMER.stem(w) for w in _BANNED_WORDS}
_LEMMATIZED_BANNED_WORDS = {lemma.strip() for w in _BANNED_WORDS for lemma in _MYSTEM.lemmatize(w)}

_REGEX_PATH = Path(__file__).parent / "regex.json"



def preprocess_stem(text, filter_stopwords=True, filter_stemmed_banned_words=True):
    """
    Предобрабатывает текст с использованием стемминга (приведения слов к их основе).
    
    Выполняет очистку текста, токенизацию, удаление стоп-слов (опционально) и стемминг,
    с возможностью фильтрации запрещенных слов после стемминга.
    
    Args:
        text (str): Исходный текст для предобработки
        filter_stopwords (bool): Флаг фильтрации стоп-слов (по умолчанию True)
        filter_stemmed_banned_words (bool): Флаг фильтрации запрещенных слов после стемминга (по умолчанию True)
    
    Returns:
        List[str]: Список обработанных слов в их стеммированной форме
    """
    cleaned = _PREPROCESS_REGEX.sub('', text.lower())
    words = word_tokenize(cleaned, language="russian")
    if filter_stopwords:
        words = [w for w in words if w not in _STOP_WORDS]
    stemmed = [_STEMMER.stem(w) for w in words]
    if filter_stemmed_banned_words:
        return [w for w in stemmed if w not in _STEMMED_BANNED_WORDS]
    return stemmed


def preprocess_lemma(text, filter_stopwords=False, filter_lemmatized_banned_words=False):
    """
    Предобрабатывает текст с использованием лемматизации (приведения слов к нормальной форме).
    
    Выполняет очистку текста, токенизацию, удаление стоп-слов (опционально) и лемматизацию,
    с возможностью фильтрации запрещенных слов после лемматизации.
    
    Args:
        text (str): Исходный текст для предобработки
        filter_stopwords (bool): Флаг фильтрации стоп-слов (по умолчанию False)
        filter_lemmatized_banned_words (bool): Флаг фильтрации запрещенных слов после лемматизации (по умолчанию False)
    
    Returns:
        List[str]: Список обработанных слов в их лемматизированной форме
    """
    cleaned = _PREPROCESS_REGEX.sub('', text.lower())
    words = word_tokenize(cleaned, language="russian")
    if filter_stopwords:
        words = [w for w in words if w not in _STOP_WORDS]
    lemmas = [_MYSTEM.lemmatize(w)[0].strip() for w in words]
    if filter_lemmatized_banned_words:
        return [w for w in lemmas if w not in _LEMMATIZED_BANNED_WORDS]
    return lemmas


class TextPreprocessor:
    """Класс для предобработки текста запросов с использованием регулярных выражений."""
    
    def __init__(self, patterns, path=_REGEX_PATH):
        """
        :param patterns: словарь вида {регулярное_выражение: замена}
        """
        self.compiled_patterns = {}
        for pattern, replacement in patterns.items():
            self.compiled_patterns[re.compile(pattern, re.IGNORECASE | re.UNICODE)] = replacement
           
        self.path = path
            
    @classmethod
    def from_file(cls, file_path=_REGEX_PATH):
        """Загружает правила из JSON-файла и создает экземпляр препроцессора."""
        with open(file_path, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        return cls(patterns, path=file_path)

    def preprocess(self, text: str) -> str:
        """Применяет все правила замены к тексту."""
        for pattern, replacement in self.compiled_patterns.items():
            text = pattern.sub(replacement, text)
        return text