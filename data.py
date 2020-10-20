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

def load_name_pairs(train_folder, feature_func=features.gen_features, distribution=default_pair_distribution):
    X, y = load_data(train_folder, feature_func)
    num_observations = len(X)
    newX = []

    directors = {}

    def modify(director):
        return Counter(dict(("other_" + key, val) for key, val in director.items()))

    directors_by_race = {}
    for race in distribution:
        logger.info(f"Generate director pairs for {race} directors...")
        directors = [x for (x, y) in zip(X, y) if y == race]
        directors_by_race[race] = directors

    merged_directors = []
    for race in distribution:
        directors = directors_by_race[race]
        co_directors_races = sample_race(race, distribution, len(directors))
        co_directors = choice(directors_by_race[race], len(directors))
        pairs = zip(directors, co_directors)

        modified_pairs = tqdm(((d, modify(c)) for d, c in pairs), "Modified pairs")
        merged_directors += tqdm([d + o for d, o in modified_pairs], f"Generating director pairs for {race} directors...")

    return merged_directors, y

    logger.info("Generate director pairs...")
    for idx, (director, race) in enumerate(zip(X, y)):
        progress = round(idx / num_observations * 100, 2)
        sys.stderr.write(f"\r{progress}%                            ")
        other_race = sample_race(race, distribution)
        other_directors = directors[other_race]
        other_director = np.random.choice(other_directors)
        other_director2 = Counter(dict(("other_" + key, val) for key, val in other_director.items()))

        newX.append(director + other_director2)

    sys.stderr.write("\r")
    return newX, y
        

    


    

