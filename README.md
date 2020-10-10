# name-classifier
Naive bayes classifier to infer a personal attribute from a name

## Run interactively
```
python classify.py interactive bayes.model
```

## Bulk classification
`FILENAME` contains one name per line

```
python classify.py classify-file bayes.model FILENAME
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

Results with the included classifier are as follows:

```
train_recall_weighted: 0.9529765642696487 +- 0.003419455885185841
test_recall_weighted: 0.9031480378961823 +- 0.015336390204211256

train_precision_weighted: 0.9531845782042195 +- 0.0032450447163395527
test_precision_weighted: 0.9077534242950783 +- 0.015690734997744416

train_f1_weighted: 0.952591256417799 +- 0.003539202683287972
test_f1_weighted: 0.9033803735927306 +- 0.015339696303063408

train_f1_macro: 0.9486126065066498 +- 0.004193425974147959
test_f1_macro: 0.8973515190888627 +- 0.01824740492251232

train_f1_micro: 0.9529765642696487 +- 0.003419455885185841
test_f1_micro: 0.9031480378961823 +- 0.015336390204211256


Confusion matrix, without normalization
[[ 6902     1    31    37   117]
 [    9  1167     3     0    26]
 [   34     8  3462   121   629]
 [   11     4    82  2205    70]
 [   55    11   243   123 10674]]
Normalized confusion matrix
[[97.38  0.01  0.44  0.52  1.65]
 [ 0.75 96.85  0.25  0.    2.16]
 [ 0.8   0.19 81.38  2.84 14.79]
 [ 0.46  0.17  3.46 92.96  2.95]
 [ 0.5   0.1   2.19  1.11 96.11]]
              precision    recall  f1-score   support

     African       0.98      0.97      0.98      7088
     Chinese       0.98      0.97      0.97      1205
    Coloured       0.91      0.81      0.86      4254
      Indian       0.89      0.93      0.91      2372
       White       0.93      0.96      0.94     11106

    accuracy                           0.94     26025
   macro avg       0.94      0.93      0.93     26025
weighted avg       0.94      0.94      0.94     26025
```


## Train a classifier
```
python train.py train TRAINING_DIR bayes.model
```


`TRAINING_DIR` should contain a list of files, one per class. Each file should be called `<classname>.train`. Each file contains one name per line
