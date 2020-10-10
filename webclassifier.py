#!.env/bin/flask

import pickle
from flask import Flask
from features import gen_features
import classify
import sys

app = Flask(__name__)
race_model = sys.argv[1]
classify = classify.get_classifier(open(race_model))
classify = classify.get_classifier(race_model, use_classify_many=False)

@app.route('/race/<name>')
def index(name):
    return classify(name)

if __name__ == '__main__':
    app.run(debug=False)
