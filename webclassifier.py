#!.env/bin/flask

import pickle
import json

from flask import Flask
from flask import Flask, request, jsonify

from features import gen_features
import classify
import sys

def jsonpify(obj):
    """
    Like jsonify but wraps result in a JSONP callback if a 'callback'
    query param is supplied.
    """
    try:
        callback = request.args['callback']
        response = app.make_response("%s(%s)" % (callback, json.dumps(obj)))
        response.mimetype = "text/javascript"
        return response
    except KeyError:
        return jsonify(obj)

app = Flask(__name__)
race_model = sys.argv[1]
#classify = classify.get_classifier(open(race_model))
classify = classify.get_classifier(race_model, use_classify_many=False)

metadata = {
    "name": "Race Classifier",
    "defaultTypes": [],
}

def classify_reconcile(name):
    return [{
        "id": "My id",
        "name": classify(name),
        "score": 100,
        "match": True,
        "type": [{"id": "/type/demographics", "name": "Race"}]
    }]

@app.route("/", methods=['POST', 'GET'])
def reconcile():
    queries = request.form.get('queries')
    if queries:
        queries = json.loads(queries)
        print(len(queries))
        results = {}
        for (key, query) in queries.items():
            results[key] = {"result": classify_reconcile(query["query"])}
        return jsonpify(results)

    return jsonpify(metadata)

@app.route('/race/<name>')
def race_lookup(name):
    return classify(name)

if __name__ == '__main__':
    app.run(debug=True)
