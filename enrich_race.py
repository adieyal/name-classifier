import sys
import logging
import json
import classify
import gzip
"""
Add race to director
"""

def open_file(filename):
    if filename.endswith(".gz"):
        return gzip.open(filename)
    else:
        return open(filename)

logger = logging.getLogger(__name__)

filename = sys.argv[1]
model = sys.argv[2]

names = set()

logger.info("Loading names...")
sys.stderr.write("Loading names...")
for row in open_file(filename):
    js = json.loads(row)
    for director in js["Director"]:
        name = director["Director Name"] + " " + director["Director Surname"]
        names.add(name)


logger.warning("Classifying names...")
sys.stderr.write("Classifying names...")

classifier = classify.get_classifier(model, use_classify_many=True)
names = list(names)
labels = classifier(names)
lookup = dict(zip(names, labels))

logger.info("Writing names to file...")
sys.stderr.write("Writing names to file...")
for row in open_file(filename):
    js = json.loads(row)
    for director in js["Director"]:
        name = director["Director Name"] + " " + director["Director Surname"]
        race = lookup.get(name, "Unknown")
        director["race"] = race
    js2 = json.dumps(js)
    sys.stdout.write(js2)
    sys.stdout.write("\n")
    sys.stderr.write(f"\r{js['Enterprise']['Enterprise Name']}")
