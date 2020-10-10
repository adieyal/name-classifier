from __future__ import division
import logging
import sys
import click
from features import gen_features
from model import load_classifier
from datetime import datetime

logger = logging.getLogger(__name__)

def get_classifier(modelfile, cache=True, use_classify_many=False):
    classifier, encoder = load_classifier(modelfile)
    cache_dict = {}
    classes = encoder.classes_

    def classify(name):
        name = name.upper()
        if name in cache_dict:
            return cache_dict[name]
        else:
            feature = gen_features(name)
            label_numeric = classifier.predict(feature)[0]
            label = classes[label_numeric]
            
            if cache:
                cache_dict[name] = label
            return label

    classify.classifier = classifier

    def classify_many(names):
        features = [gen_features(name) for name in names]
        labels = classifier.predict(features)
        return [classes[l] for l in labels]

    classify_many.classifier = classifier

    if use_classify_many:
        return classify_many
    else:
        return classify

@click.group()
def cli():
    pass

@cli.command()
@click.argument("model", type=click.File("rb"))
@click.argument("name")
def classify_name(model, name):
    classify = get_classifier(model)
    print(classify(name))

def chunks(iter, size):
    accum = []
    for el in iter:
        accum.append(el.strip())
        if len(accum) == size:
            yield accum
            accum = []
    if len(accum) > 0:
        yield accum

@cli.command()
@click.argument("model", type=click.File("rb"))
@click.argument("filename", type=click.File("r", encoding="latin1"))
def classify_file(model, filename, chunk_size=5000):
    logger.info("Loading classifier")
    now = datetime.now()
    classify = get_classifier(model, use_classify_many=True)
    logger.info("Loaded classifier: %d seconds" % (datetime.now() - now).seconds)
    for chunk in chunks(filename, chunk_size):
        start = datetime.now()
        labels = classify(chunk)
        end = datetime.now()
        delta = end - start
        diff = delta.seconds * chunk_size + delta.microseconds / chunk_size

        per_sec = chunk_size/diff * chunk_size
        sys.stderr.write("\r%d names classified per second" % per_sec)
        sys.stderr.flush()
        for n, l in zip(chunk, labels):
            s = "%s,%s" % (n, l)
            print(s)

@cli.command()
@click.argument("model", type=click.File("rb"))
def interactive(model):
    classify = get_classifier(model, use_classify_many=False)
    while True:
        text = input("Name (or q to quit): ")
        if text.lower().strip() == "q":
            break
        print(classify(text))

if __name__ == "__main__":
    cli()
