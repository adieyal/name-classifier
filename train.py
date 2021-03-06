

import click
import glob
import logging
import os

import plotting

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate, learning_curve, train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

import features
import model
import data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

feature_generators = {
    "gen_features" : features.gen_features,
    "gen_features_with_language" : features.gen_features_with_language,
}

data_loaders = {
    "basic_loader" : data.load_data,
    "name_pairs_loader" : data.load_name_pairs,
}

def get_feature(feature_generator):
    if feature_generator in feature_generators:
        feature_func = feature_generators[feature_generator]
    else:
        raise Exception("Unknown feature generator: %s, expected one of:\n%s" % (feature_generator, "\n".join(feature_generators)))
    return feature_func

def get_data_loader(data_loader):
    if data_loader in data_loaders:
        return data_loaders[data_loader]
    else:
        raise Exception("Unknown data loader: %s, expected one of:\n%s" % (data_loader, "\n".join(data_loaders)))

def get_classifier():
    pipeline = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier', MultinomialNB()),
    ])
    return pipeline

@click.group()
def cli():
    pass

def output_errors(X, y_pred, y_test):
    for x, y_p, y_t in zip(X, y_pred, y_test):
        if y_p != y_t:
            name = [k for k in x.keys() if k.startswith("name_")][0]
            print(name, y_p, y_t)

@cli.command()
@click.argument("training_dir")
@click.argument("modelfile", type=click.File("wb"))
@click.option("--feature_generator", default="gen_features", help="feature generator to use")
def train(training_dir, modelfile, feature_generator):
    encoder = LabelEncoder()

    feature_func = get_feature(feature_generator)
    X, y = data.load_data(training_dir, feature_func)

    y_transformed = encoder.fit_transform(y)

    classifier = get_classifier()
    logger.info("Training classifier")
    classifier.fit(X, y_transformed)
    logger.info("Saving classifier")
    model.save_classifier([classifier, encoder], modelfile)

@cli.command()
@click.argument("training_dir")
@click.option('--training-curve/--no-training-curve', default=False, help="Create training curves")
@click.option('--cross-validation/--no-cross-validation', default=False, help="Run crossvalidation")
@click.option('--confusion-matrix/--no-confusion-matrix', default=False, help="Create a confusion matrix")
@click.option('--print-classification-report/--no-print-classification-report', default=False, help="Print out a classification report")
@click.option("--njobs", default=1, help="Number of parallel processes to use")
@click.option("--feature_generator", default="gen_features", help="feature generator to use")
@click.option("--data-loader", default="basic_loader", help="data loading mechansim")
@click.option("--print-misclassifications/--no-print-misclassifications", default=False, help="Print misclassified directors")
def test(training_dir, njobs, training_curve, cross_validation, confusion_matrix, print_classification_report, feature_generator, data_loader, print_misclassifications):

    feature_func = get_feature(feature_generator)
    data_loader_func = get_data_loader(data_loader)
    encoder = LabelEncoder()

    classifier = get_classifier()

    X, y = data_loader_func(training_dir, feature_func=feature_func)

    logger.info("Transforming")
    y_transformed = encoder.fit_transform(y)
    class_labels = encoder.classes_

    if training_curve:
        logger.info("Training curve")
        d = DictVectorizer(sparse=True)
        plotting.plt_training_curve(MultinomialNB(), d.fit_transform(X), encoder.fit_transform(y), njobs)


    if cross_validation:
        logger.info("Cross validation")
        cv = StratifiedKFold(n_splits=5)

        scoring = ["recall_weighted", "precision_weighted", "f1_weighted", "f1_macro", "f1_micro"]
        scores = cross_validate(classifier, X, y_transformed, scoring=scoring, cv=cv, n_jobs=njobs, return_train_score=True)

        def print_score(scores, metric):
            metric_key = "train_%s" % metric
            vals = scores[metric_key]
            print("%s: %s +- %s" % (metric_key, vals.mean(), vals.std()))

            metric_key = "test_%s" % metric
            vals = scores[metric_key]
            print("%s: %s +- %s" % (metric_key, vals.mean(), vals.std()))
            print("")
        for m in scoring:
            print_score(scores, m)
        print("\a")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed)
    logger.info("Fitting")
    
    classifier.fit(X_train, y_train)

    logger.info("Predicting")
    y_pred = classifier.predict(X_test)

    if confusion_matrix:
        logger.info("Confusion matrix")
        plotting.print_confusion_matrix(y_test, y_pred, class_labels)

    if print_classification_report:
        logger.info("Classification report")
        print(classification_report(y_test, y_pred, target_names=class_labels))

    if print_misclassifications:
        output_errors(X_test, y_pred, y_test)


if __name__ == "__main__":
    cli()
