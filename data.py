import os
import glob
import logging
import features

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

