import joblib

def save_classifier(classifier, modelfile):
    joblib.dump(classifier, modelfile)

def load_classifier(modelfile):
    classifier = joblib.load(modelfile)
    return classifier
