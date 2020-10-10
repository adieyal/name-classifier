# name-classifier
Naive bayes classifier to infer a personal attribute from a name

## Run interactively
```
python classify.py interactive bayes.model
```

## Run in a web browser
```
.env/bin/python3 webclassifier.py bayes.model

curl http://localhost:5000/race/Nelson%20Mandela
```

## Testing classifiers
```
python train.py test \
    --cross-validation \
    --confusion-matrix \
    --print-classification-report \
    --training-curve \
    TRAINING_DIR
```


## Train a classifier
```
python train.py train TRAINING_DIR bayes.model
```


TRAINING_DIR should contain a list of files, one per class. Each file should be called <classname>.train. Each file contains one name per line
