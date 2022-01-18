from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report
from modules import *
from utils import *
X_train, X_val, X_test, y_train, y_val, y_test = load_datasets()

#densify sparse matricies
x_tr = X_train.toarray()
x_va = X_val.toarray()
# clf = DummyClassifier(strategy="uniform", random_state=42)
clf = RandomForestClassifier(n_estimators = 800, max_depth = 15, min_samples_split=15, n_jobs=-1, random_state=42, verbose=1)
clf.fit(x_tr, y_train)
print("ROC_AUC score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
print("ROC_AUC score for test set: {:.4f}.\n".format(predict_labels(clf, X_test, y_test)))
print(f'Classification Report: \n{classification_report(y_test, clf.predict(X_test), labels=[0,1])}')

clf2 = AdaBoostClassifier(base_estimator=clf, n_estimators=400, random_state = RAN_STATE)
clf2.fit(X_train, y_train)
# Print the results of prediction for both training and testing
print("ROC_AUC score for training set: {:.4f}.".format(predict_labels(clf2, X_train, y_train)))
print("ROC_AUC score for test set: {:.4f}.\n".format(predict_labels(clf2, X_test, y_test)))
print(f'Classification Report: \n{classification_report(y_test, clf2.predict(X_test), labels=[0,1])}')