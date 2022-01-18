from modules import load_datasets
from utils import FIG_SIZE, RAN_STATE
import numpy as np

import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Parameters
sdg_params = dict(alpha=1e-5, penalty="l2", loss="log")

# Supervised Pipeline
pipeline = Pipeline(
    [
        ("clf", SGDClassifier(**sdg_params)),
    ]
)
# SelfTraining Pipeline
st_pipeline = Pipeline(
    [
        ("clf", SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True)),
    ]
)
max = 0

def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test , isSSL):
    num_unlabeled_data = sum(1 for x in y_train if x == -1)
    print("Number of training samples:", len(X_train))
    print("Unlabeled samples in training set:", num_unlabeled_data)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_sco = f1_score(y_test, y_pred)
    print(
        "Micro-averaged F1 score on test set: %0.3f"
        % f1_sco
    )

    global max
    if (f1_sco > max) and (isSSL) :
        max = f1_sco
        print(f'Classification Report: on Train \n{classification_report(y_train, clf.predict(X_train), labels=[0,1])}')
        print(f'Classification Report on Val: \n{classification_report(y_test, y_pred, labels=[0,1])}')
        pickle.dump(clf, open("SSL_SGD.pkl", "wb"))

    print("-" * 10)
    print()
    return f1_sco


if __name__ == "__main__":

    X_train, _ , X_test, y_train, _ , y_test = load_datasets()

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    print("Supervised SGDClassifier on 100% of the data:")
    eval_and_print_metrics(pipeline, X_train, y_train, X_test, y_test, False)

    supervised_score, semi_supervised_score = [], []

    masks = np.linspace(0.02, 0.6, num=25)
    for mask in masks:
        X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, train_size=mask, random_state=RAN_STATE, stratify=y_train)

        print(f"Supervised SGDClassifier on {mask*100}% of the training data:")
        supervised_score.append(eval_and_print_metrics(pipeline, X_tr, y_tr, X_test, y_test, False))

        np.asarray(y_te).fill(-1)
        X = np.concatenate((X_tr, X_te), axis=0)
        y = np.concatenate((y_tr, y_te), axis=0)        

        print(f"SelfTrainingClassifier on {mask*100}% of the training data (rest is unlabeled):")
        semi_supervised_score.append(eval_and_print_metrics(st_pipeline, X, y, X_test, y_test, True))
    
    plt.figure(figsize = FIG_SIZE)
    plt.title('Number of Labeled data points vs Micro-averaged F1 score')
    plt.grid()
    plt.xlabel('Mask Ratio = (Number of labeled data points)/(total data points)')
    plt.ylabel('Weighted-averaged F1 score')
    plt.plot(masks, supervised_score, label='Supervised')
    plt.plot(masks, semi_supervised_score, label='Semi-Supervised')
    plt.legend(loc="upper right")
    plt.show()    