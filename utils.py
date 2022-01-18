import pickle
import os
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,StratifiedKFold

# Import the supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

RAN_STATE = 42
FIG_SIZE = (14, 8)

scoring = {"AUC": "roc_auc", "f1": "f1"}

grid = {
    
    "LogRegression" : {
        "params" : {'C': [10**i for i in range(-5,5)], 
                    'class_weight': [None, 'balanced'], 
                    'penalty':['l1','l2']},
        "estimator": LogisticRegression(random_state = RAN_STATE, n_jobs=-1)
    },
    "GaussianNB" : {
        "params" : {},
        "estimator": GaussianNB()
    },
    "DecisionTree" : { 
        "params" : {'max_features': ['auto', 'sqrt', 'log2'],
                    'ccp_alpha': [0.1, .01, .001],
                    'max_depth' : [3, 6, 9, 12, 15],
                    'criterion' :['gini', 'entropy']},
        "estimator": DecisionTreeClassifier(random_state = RAN_STATE) 
    },
    "RandomForest": {
        "params": {'n_estimators': [100, 200, 300], 
                    'max_depth': [5, 10, 18],  
                    'min_samples_split': [2, 5, 10]},
                    # 'min_samples_leaf': [1, 4, 8]},
        "estimator": RandomForestClassifier(n_jobs=-1)
    },
    "AdaBoost": {
        "params": {'n_estimators' : [30, 50, 70, 80],
                    'algorithm': ['SAMME'],
                    'learning_rate': [0.001, 0.01, .80, 1.0]}, 
        "estimator": AdaBoostClassifier(random_state = RAN_STATE),
    },
    "SGD": {
        "estimator" : SGDClassifier(),
        "params": {
            "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
            "alpha" : [0.0001, 0.001, 0.01, 0.1],
            "penalty" : ["l2", "l1", "none"],
        }
    }
}

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    return clf

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on roc_auc score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    probas = clf.predict_proba(features)
    end = time()
    
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    return roc_auc_score(target.values, probas[:,1].T)

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on roc_auc score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, X_train.shape[0]))
    
    # Train the classifier
    clf = train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print("ROC_AUC score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("ROC_AUC score for test set: {:.4f}.\n".format(predict_labels(clf, X_test, y_test)))
    print(f'Classification Report: \n{classification_report(y_test, clf.predict(X_test), labels=[0,1])}')
    
def clf_test_roc_score(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    return roc_auc_score(y_test, probas[:,1].T)

def hypothesis_complexity_study(x_tr, x_te, y_test, y_train):
    # Initialize the models using a random state were applicable.
    clf_list = []
    for key, _ in grid.items():
        print(f'Fitting grid search for {key}')
        if not os.path.exists("models/"+key+".pkl"):
            print(f'Model does not exists for {key}. Please run grid search to save best tuned model.')
        else:
            print(f'Model exists. Loading model for {key}')
            clf_list.append(pickle.load(open("models/"+key+".pkl", 'rb')))
            
    # Set up the training set sizes for 100, 200 and 300 respectively.
    train_feature_list = [x_tr[0:10000],x_tr[0:20000], x_tr[0:40000],x_tr]
    train_target_list = [y_train[0:10000], y_train[0:20000], y_train[0:40000], y_train]


    # Execute the 'train_predict' function for each of the classifiers and each training set size
    for clf in clf_list:
        for a, b in zip(train_feature_list, train_target_list):
            train_predict(clf, a, b, x_te, y_test)

    ### Visualize all of the classifiers    
    plt.figure(figsize=FIG_SIZE)                                                            
    for clf in clf_list:
        x_graph = []
        y_graph = []
        for a, b in zip(train_feature_list, train_target_list):
            y_graph.append(clf_test_roc_score(clf, a, b, x_te, y_test))
            x_graph.append(len(a))
        plt.scatter(x_graph,y_graph)
        plt.plot(x_graph,y_graph, label = clf.__class__.__name__)
    plt.title('Comparison of Different Classifiers')
    plt.xlabel('Training Set Size')
    plt.ylabel('ROC_AUC score on Test Set')
    plt.legend(loc="upper right")
    plt.show()

def plot_cv_results(gs):

    print(gs.cv_results_)
    plt.figure(figsize=FIG_SIZE)  
    plt.xlabel('Index')
    plt.ylabel('CV scores')
    plt.plot(gs.cv_results_['mean_test_AUC'], label='AUC Validation')
    plt.plot(gs.cv_results_['mean_train_AUC'], label='AUC Train')
    plt.plot(gs.cv_results_['mean_test_f1'], label='F1 Validation')
    plt.plot(gs.cv_results_['mean_train_f1'], label='F1 Train')
    plt.legend(loc='best')
    plt.show()

def grid_search(X_train, y_train, X_val, y_val):
    probas_dict = {}
    for key, value in grid.items():
        print(f'Fitting grid search for {key}')
        if not os.path.exists(key+".pkl"):
            print(f'For key {key}')
            gs = GridSearchCV(
                estimator=value["estimator"],
                param_grid=value["params"],
                cv=StratifiedKFold(),
                scoring=scoring,
                refit="AUC",
                verbose=2,
                return_train_score=True
            )

            gs.fit(X_train, y_train)
            plot_cv_results(gs)
            clf = gs.best_estimator_
            print(clf)
            print(gs.best_params_)
            print(gs.best_score_)
            pickle.dump(clf, open(key+".pkl", "wb"))
        else:
            print('Model already exists. Loading model')
            clf = pickle.load(open(key+".pkl", 'rb'))
        probas_train = clf.predict_proba(X_train)
        probas = clf.predict_proba(X_val)
        probas_dict[key] = probas
        print('Best ROC_AUC Score on Training Set:',roc_auc_score(y_train, probas_train[:,1].T))
        print('Best ROC_AUC Score on Validation Set:',roc_auc_score(y_val, probas[:,1].T))
        print(f'Classification Report: on Train \n{classification_report(y_train, clf.predict(X_train), labels=[0,1])}')
        print(f'Classification Report on Val: \n{classification_report(y_val, clf.predict(X_val), labels=[0,1])}')
        print (f'Grid search completed for {key}')

    return probas_dict

def print_roc_curve(y_test, grid):

    plt.figure(figsize = FIG_SIZE)
    plt.title('ROC Curve for Helpfulness Rating')
    plt.grid()
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    for key, value in grid.items():
        probas = value
        plt.plot(roc_curve(y_test, probas[:,1])[0], roc_curve(y_test, probas[:,1])[1], label=key)
    plt.legend(loc="upper right")
    plt.show()    

def model_evaluation(X_train, y_train, X_test, y_test):
    probas_dict = {}
    for key, _ in grid.items():
        if not os.path.exists(key+".pkl"):
            print(f'Model does not exists for {key}. Please run grid search to save best tuned model.')
        else:
            print("--------------------------------")
            print(f'Model exists. Loading model for {key}')
            clf = pickle.load(open("models/"+key+".pkl", 'rb'))
            probas = clf.predict_proba(X_test)
            probas_dict[key] = probas
            y_pred = clf.predict(X_test)
            print("ROC_AUC score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
            print("ROC_AUC score for test set: {:.4f}.\n".format(predict_labels(clf, X_test, y_test)))
            print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred)}')
            print(f'Classification Report: \n{classification_report(y_test, clf.predict(X_test), labels=[0,1])}')
            print("--------------------------------")

    return probas_dict

def evaluate_ssl_model(X_train, y_train, X_test, y_test):

    print("--------------------------------")
    print(f'Model exists. Loading model for SSL')
    clf = pickle.load(open("models/SSL_SGD.pkl", 'rb'))
    y_pred = clf.predict(X_test)
    print("ROC_AUC score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("ROC_AUC score for test set: {:.4f}.\n".format(predict_labels(clf, X_test, y_test)))
    print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification Report: \n{classification_report(y_test, clf.predict(X_test), labels=[0,1])}')
    print("--------------------------------")