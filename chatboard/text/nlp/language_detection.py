from langdetect import detect
import iso639

import time


def get_language(text, return_code=False):
    '''Detect language of text.'''
    start_time = time.time()
    lang_code = detect(text)
    detaction_time = time.time()
    if return_code:
        return lang_code
    
    language = iso639.languages.part1.get(lang_code, None)
    language = language.name if language else "English"
    language_time = time.time()
    # print(f"Language detection time: {detaction_time - start_time}")
    # print(f"Language display time: {language_time - detaction_time}")
    return language