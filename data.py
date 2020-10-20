import os
import sys
from collections import Counter
import glob
import logging
import features
import random
import numpy as np
from tqdm import tqdm
from numpy.random import choice

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def load_data(train_folder, feature_func=features.gen_features):
    data = {}
    X = []
    y = []

    for fn in glob.glob("%s/*.train" % train_folder):
        ethnicity = os.path.splitext(os.path.basename(fn))[0]
        logger.info("Generating features for: %s" % ethnicity)
        with open(fn) as fp:
            dataset = [n.strip() for n in fp]
            logger.info("%s contains %d examples" % (fn, len(dataset)))
        for n in dataset:
            try:
                X.append(feature_func(n))
                y.append(ethnicity)
            except features.FeatureException:
                logger.warn("Ignored %s" % n)
    return X, y

default_pair_distribution = {
    "African": { "African": 0.937, "Coloured": 0.016, "East Asian": 0, "South Asian": 0.009, "White": 0.038, },
    "Coloured": { "African": 0.168, "Coloured": 0.459, "East Asian": 0.001, "South Asian": 0.046, "White": 0.326, },
    "East Asian": { "African": 0.126, "Coloured": 0.026, "East Asian": 0.654, "South Asian": 0.049, "White": 0.145, },
    "South Asian": { "African": 0.12, "Coloured": 0.057, "East Asian": 0.002, "South Asian": 0.698, "White": 0.123, },
    "White": { "African": 0.114, "Coloured": 0.087, "East Asian": 0.002, "South Asian": 0.026, "White": 0.771, },
}

def sample_race(race, distribution, num_samples):
    d = distribution[race]
    return choice(list(d.keys()), num_samples, p=list(d.values()))

import random
def pick_director(directors):
    rand_idx = random.randint(0, len(directors) - 1)
    return directors[rand_idx]

def modify(director):
    return Counter(dict(("other_" + key, val) for key, val in director.items()))
    

def load_name_pairs(train_folder, feature_func=features.gen_features, distribution=default_pair_distribution):
    X, y = load_data(train_folder, feature_func)
    newX = []
    newY = []

    directors = {}


    directors_by_race = {}
    for race in distribution:
        directors = [x for (x, y) in zip(X, y) if y == race]
        directors_by_race[race] = directors

    merged_directors = []
    for race in distribution:
        directors = directors_by_race[race]
        co_directors_races = sample_race(race, distribution, len(directors))
        co_directors = [pick_director(directors_by_race[co]) for co in co_directors_races]
        pairs = zip(directors, co_directors)

        modified_pairs = ((d, modify(c)) for d, c in pairs)
        newX += tqdm((d + o for d, o in modified_pairs), f"Generating director pairs for {race} directors...")
        newY += [race] * len(directors)

    return newX, newY

    


    

