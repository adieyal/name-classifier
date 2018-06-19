import pandas as pd
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plt_training_curve(estimator, X, y, njobs=1, train_sizes=None, title="Learning Curves"):
    train_sizes = train_sizes or [5, 100, 500, 2000, 5000, 10000, 30000, 65138]
    train_sizes = [5, 100, 500, 2000, 5000, 10000, 25000, 50000, 76000]
    #train_sizes = [size for size in train_sizes if size <= X.shape[0]]

    logger.info("Training curves")
    train_sizes, train_scores, validation_scores = learning_curve(
           estimator=estimator,
           X=X,
           y=y,
           train_sizes=train_sizes,
           verbose=1,
           shuffle=True,
           cv=5,
           exploit_incremental_learning=False,
           n_jobs=njobs,
           scoring='accuracy'
    )

    print('Training scores:\n\n', train_scores)
    print('\n', '-' * 70) # separator to make the output easy to read
    print('\nValidation scores:\n\n', validation_scores)

    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)

    print('Mean training scores\n\n', pd.Series(train_scores_mean, index=train_sizes))
    print('\n', '-' * 20) # separator
    print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index=train_sizes))

    plt.style.use('seaborn')
    plt.figure

    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')

    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)
    plt.show()

def print_confusion_matrix(y_test, y_pred, class_labels):
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix, without normalization')
    print(cm)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print('Normalized confusion matrix')
    print(cm_normalized)

    plt.figure
    plot_confusion_matrix(cm_normalized, class_labels, title='Normalized confusion matrix')
    plt.show()
