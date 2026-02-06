import time

import iso639
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def get_language(text, return_code=False):
    '''Detect language of text.'''
    start_time = time.time()
    try:
        lang_code = detect(text)
    except LangDetectException as e:        
        return None
    detaction_time = time.time()
    if return_code:
        return lang_code
    
    language = iso639.languages.part1.get(lang_code, None)
    language = language.name if language else "English"
    language_time = time.time()    
    return language