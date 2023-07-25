import time
import functools
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def timer(func):
    '''
    Decorator to log time taken by classifier
    '''
    # from https://realpython.com/primer-on-python-decorators/

    # print runtime of decorated function
    @functools.wraps(func)

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time    
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    
    return wrapper_timer


@timer
def run_linear_svm_classification(X, y, X_test, y_test, scoring_measure, random_seed):

    '''This part of the code was partly inspired by Sandra Vieira, Rafael Garcia-Dias, Walter Hugo Lopez Pinaya - Chapter 19: A Step-By-Step Tutorial On How To Build A Machine Learning Model (https://github.com/MLMH-Lab/How-To-Build-A-Machine-Learning-Model/)'''


    # scaling
    scaler = StandardScaler()

    # X and X_test are images, convert to flat matrices
    X_flat = list(X)
    X_flat = np.nan_to_num(X_flat)
    scaler.fit(X_flat)
    X_flat = np.nan_to_num(scaler.transform(X_flat))

    X_test_flat = list(X_test)
    X_test_flat = np.nan_to_num(X_test_flat)
    X_test_flat = np.nan_to_num(scaler.transform(X_test_flat))

    y = y.astype('category').cat.codes.values 
    y_test = y_test.astype('category').cat.codes.values
    
    clf = svm.LinearSVC(random_state=random_seed, max_iter= 1000000) 

    param_grid = {'C': [2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 10, 25, 50, 100], 'penalty' : ['l2']} 

    # grid search
    internal_cv = StratifiedKFold(n_splits=2)
    grid_cv = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=internal_cv,
                            scoring=scoring_measure,
                            verbose=1)
    
    grid_result = grid_cv.fit(X_flat, y)

    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with: %r' % (mean, stdev, param))

    print("Best estimator: ", grid_cv.best_estimator_)

    best_clf = grid_cv.best_estimator_

    prediction = best_clf.predict(X_test_flat)

    print('Confusion matrix')
    cm = confusion_matrix(y_test, prediction)
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    bac_test = metrics.balanced_accuracy_score(y_test, prediction)
    sens_test = tp / (tp + fn)
    spec_test = tn / (tn + fp)

    print('Balanced accuracy: %.3f ' % bac_test)
    print('Sensitivity: %.3f ' % sens_test)
    print('Specificity: %.3f ' % spec_test)

    coefs = best_clf.coef_

    return prediction, bac_test, sens_test, spec_test, cm, coefs

@timer
def run_nonlinear_svm_classification(X, y, X_test, y_test, scoring_measure, random_seed):

    # scaling
    scaler = StandardScaler()

    # X and X_test are images, convert to flat matrices
    X_flat = list(X)
    X_flat = np.nan_to_num(X_flat)
    scaler.fit(X_flat)
    X_flat = np.nan_to_num(scaler.transform(X_flat))

    X_test_flat = list(X_test)
    X_test_flat = np.nan_to_num(X_test_flat)
    X_test_flat = np.nan_to_num(scaler.transform(X_test_flat))

    y = y.astype('category').cat.codes.values 
    y_test = y_test.astype('category').cat.codes.values
    
    clf = svm.SVC(kernel = 'poly', random_state=random_seed, max_iter= 1000000) 

    param_grid = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1, 10, 25, 50, 100]} 

    # grid search
    internal_cv = StratifiedKFold(n_splits=2)
    grid_cv = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=internal_cv,
                            scoring=scoring_measure,
                            verbose=1)
    
    grid_result = grid_cv.fit(X_flat, y)

    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with: %r' % (mean, stdev, param))

    print("Best estimator: ", grid_cv.best_estimator_)

    best_clf = grid_cv.best_estimator_

    prediction = best_clf.predict(X_test_flat)

    print('Confusion matrix')
    cm = confusion_matrix(y_test, prediction)
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    bac_test = metrics.balanced_accuracy_score(y_test, prediction)
    sens_test = tp / (tp + fn)
    spec_test = tn / (tn + fp)

    print('Balanced accuracy: %.3f ' % bac_test)
    print('Sensitivity: %.3f ' % sens_test)
    print('Specificity: %.3f ' % spec_test)

    coefs = best_clf.coef_

    return prediction, bac_test, sens_test, spec_test, cm, coefs


@timer
def run_rf_classification(X, y, X_test, y_test, scoring_measure, random_seed):

    ''' This part of the code was partly inspired by Ruben Otter - Unsupervised Gaze Event Discrimination (https://github.com/LEO-UMCG/Unsupervised-Gaze-Event-Discrimination/)'''

   # scaling to variance but keep sparsity structure of data
    scaler = StandardScaler()

    # X and X_test are images, convert to flat matrices
    X_flat = list(X)
    X_flat = np.nan_to_num(X_flat)
    scaler.fit(X_flat)
    X_flat = np.nan_to_num(scaler.transform(X_flat))

    X_test_flat = list(X_test)
    X_test_flat = np.nan_to_num(X_test_flat)
    X_test_flat = np.nan_to_num(scaler.transform(X_test_flat))
    
    y = y.astype('category').cat.codes.values 
    y_test = y_test.astype('category').cat.codes.values
    
    # number of trees
    n_estimators = [int(x) for x in np.linspace(start = 40, stop = 2500, num = 40)] 
    criterion = ['gini', 'entropy', 'log_loss']
    # number of features to take into account at every split
    max_features = ['log2', 'sqrt']
    # max number of levels in one tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 20)]
    max_depth.append(None)
    # min samples needed to split a node
    min_samples_split = [2, 3, 4, 5, 10]
    # min samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4]
    # method of selecting samples for training each tree
    bootstrap = [True, False]
    
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion': criterion}
    print(random_grid)

    # random forest classifier using Scikit-learn package
    clf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=random_seed, n_jobs = -1)
    rf_random.fit(X_flat, y)
    
    print("train arrays img", X_flat[:10])
    print("train lab", y)

    print(rf_random.best_estimator_)
    print(rf_random.best_params_)

    rf_classifier = rf_random.best_estimator_

    rf_classifier.fit(X_flat, y)

    y_pred = rf_classifier.predict(X_test_flat)

    print("accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("balanced accuracy:", metrics.balanced_accuracy_score(y_test, y_pred))
    print("f1:", metrics.f1_score(y_test, y_pred))
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).ravel()
    tn, fp, fn, tp = conf_matrix
    specificity = tn / (tn+fp)

    print(classification_report(y_test, y_pred))

    coefs = rf_classifier.feature_importances_
    print(coefs)

    return rf_classifier, y_pred, accuracy, sensitivity, specificity, conf_matrix, coefs



