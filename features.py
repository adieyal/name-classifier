import logging
import random
import os
from glob import glob
from collections import Counter
import re

logger = logging.getLogger(__name__)

def split_name_parts(name):
    surname_parts = set([
        "VERE","VON","VAN","DE","DEL","DELLA","DI",
        "DA","PIETRO","VANDEN","DU","ST.","ST","LA",
        "TER", "VANDER", "JANSE"
    ])

    name = name.upper()
    parts = name.split()

    surname_idx = [idx for (idx, part) in enumerate(parts) if part in surname_parts]
    if len(surname_idx) == 0:
        surname = parts[-1:]
        name = parts[0:-1]
    else:
        split_idx = surname_idx[0]
        surname = parts[split_idx:]
        name = parts[0:split_idx]

    return {
        "name" : name,
        "surname" : surname
    }

def gen_ngrams(name, n):
    ngrams = []
    for name_part in name.split():
        word_ngrams = [name_part[i:i + n] for i in range(len(name_part) - n + 1)]
        if len(name_part) >= n:
            first = "b_" + word_ngrams[0]
            last = "l_" + word_ngrams[-1]
            word_ngrams += [first, last]
        ngrams += word_ngrams
    return ngrams

class FeatureException(Exception):
    pass

def gen_features_with_language(datum):
    parts = datum.split(",")
    if len(parts) != 2:
        raise FeatureException("Expected two fields: name, language. Received %s" % datum)
    name, language = parts
    features = gen_features2(name)
    features[language] = 1
    return features

def gen_features(name, include_name_part=True, include_bigrams=True, include_trigrams=True, include_4grams=True, include_surname=True, include_fullname=True):

    def generate_ngrams(token):
        features = []
        if include_bigrams:
            if len(token) >= 2:
                total_bigrams = gen_ngrams(token, 2)
                if len(total_bigrams) < 3:
                    features.append("length_short")
                features.extend(total_bigrams)
        if include_trigrams:
            if len(token) >= 3:
                total_trigrams = gen_ngrams(token, 3)
                features.extend(total_trigrams)
        if include_4grams:
            if len(token) >= 4:
                total_quadgrams = gen_ngrams(token, 4)
                features.extend(total_quadgrams)
        return features

    name = name.upper()
    features = []
    for n in name.split():
        features.extend(generate_ngrams(n))
        if include_name_part:
            features.append("namepart_" + n)

        try:
            if include_surname:
                parts = split_name_parts(name)
                surname = parts["surname"]
                surname = "_".join(surname)
                #surname = name.split()[-1]
                surname_features = generate_ngrams(surname)
                features += ["s_" + f for f in surname_features]
                features.append("surname_" + surname)
        except IndexError:
            logger.warn("No surname found: %s" % name)

    if include_fullname:
        features.append("name_" + name.replace(" ", "_"))

    return Counter(features)
