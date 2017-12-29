from collections import defaultdict
from pprint import pprint
from time import time
import sys
import copy
import itertools
import numpy as np
from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

sys.path.append("../tools/")
from tester import test_classifier


def _get_num_of_poi(data):
    """
    :param data: list of dicts with person data
    :return: number of poi
    """
    is_poi = [person.get('poi') for person in data]
    num_poi = is_poi.count(True)
    return num_poi

def _get_features():
    features_list = ['poi',
                     'bonus',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'email_address',
                     'exercised_stock_options',
                     'expenses',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'loan_advances',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'shared_receipt_with_poi',
                     'to_messages',
                     'total_payments',
                     'total_stock_value']
    features_list.remove('email_address')
    return features_list

def _order_features_by_score():
    """
    sorts the features by scores
    scores: F value of Analysis of Variance (ANOVA)
    :return: sorted features and scores
    """
    features = _get_features()
    features.remove('poi')
    scores = [10.09863783, 0.8831138, 1.19555209, 1.37250905, 34.73257743,
              3.34279872, 0.05916248, 0.75033658, 2.16001363, 8.17870326,
              7.54929549, 4.85790445, 15.21368276, 0.07779663, 10.22957239,
              1.38336448, 0.57357048, 8.82481433, 33.31087215]
    feature_score_tuple = zip(features, scores)
    feature_score_tuple_sorted = sorted(feature_score_tuple, key=lambda x: x[1], reverse=True)
    features_sorted, scores_sorted = zip(*feature_score_tuple_sorted)
    return ['poi'] + list(features_sorted), list(scores_sorted)

def _create_new_features(features):
    """
    replaces the first feature by the product of the first two features
    input: [[feat_1_sample_1, feat_2_sample_1, ..., feat_n_sample_1],
            [feat_1_sample_2, feat_2_sample_2, ..., feat_n_sample_2], ...]

    :param features:
    :return: features
    """
    num_feat = len(features)
    for vector, i in zip(features, range(num_feat)):
        vector[0] = vector[0]*vector[1]
        features[i] = vector
    return features

def _check_k_best_scores(features, labels):
    k_best = SelectKBest(k='all')
    k_best.fit(features, labels)
    print('SelectKBest scores: {0}'.format(k_best.scores_))

def _remove_outlier(data_dict):
    outlier_list = ['TOTAL']
    data_dict_copy = copy.deepcopy(data_dict)
    for name in list(data_dict_copy.keys()):
        if name in outlier_list:
            data_dict_copy.pop(name)
    return data_dict_copy

def _scale_data(features):
    myfeatures = []
    for feature in features:
        array_min = feature.min()
        array_max = feature.max()
        feature = (feature-array_min)/(array_max-array_min)
        myfeatures.append(feature)
    return np.array(myfeatures)

def _select_features(key):
    feat_select = {'pca': PCA(n_components=5),
                   'var_threshold': VarianceThreshold(threshold = 0.1),
                   'k_best': SelectKBest(k=5)}
    return feat_select[key]

def _get_classifier(key):
    clf_dict ={'svm': svm.SVC(kernel='rbf'),
               'ada_boost': AdaBoostClassifier(),
               'nb': GaussianNB(),
               'lin_reg': linear_model.LinearRegression()}
    return clf_dict[key]

def _get_train_test_data(features, labels):
    feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size = 0.4, random_state = 0)
    return feature_train, feature_test, label_train, label_test


def _cross_validate(pipeline, features, labels):
    """
     precision = Tp/(Tp + Fp)
     recall = Tp/(Tp + Fn)
    """
    print('My pipeline: {0}'.format(pipeline))
    scoring = ['precision', 'recall', 'accuracy']
    sss = StratifiedShuffleSplit(n_splits=50, test_size=0.25, random_state=42)
    scores = cross_validate(estimator=pipeline,
                            X=features,
                            y=labels,
                            scoring=scoring,
                            verbose=1,
                            cv=sss,
                            return_train_score='warn')
    print(scores.keys())
    train_recall = _get_mean_and_std(scores['train_recall'])
    test_recall = _get_mean_and_std(scores['test_recall'])
    train_precision = _get_mean_and_std(scores['train_precision'])
    test_precision = _get_mean_and_std(scores['test_precision'])
    accuracy = _get_mean_and_std(scores['test_accuracy'])
    print('train_recall: {0:0.3f} +/- {1:0.3f}'.format(train_recall[0], train_recall[1]))
    print('test_recall: {0:0.3f} +/- {1:0.3f}'.format(test_recall[0], test_recall[1]))
    print('train_precision: {0:0.3f} +/- {1:0.3f}'.format(train_precision[0], train_precision[1]))
    print('test_precision: {0:0.3f} +/- {1:0.3f}'.format(test_precision[0], test_precision[1]))
    print('accuracy: {0:0.3f} +/- {1:0.3f}'.format(accuracy[0], accuracy[1]))


def _get_mean_and_std(array):
    mean = array.mean()
    std = array.std()
    return mean, std

def _get_parameters(feat_select, clf, n):
    """
    returns tuple of parameters for parameter scan
    :param feat_select: flag
    :param clf: classifier
    :param n: number of features
    :return:
    """
    parameters = {}
    if feat_select == 'pca':
        parameters['dim_reduct__n_components'] = tuple(range(1, n+1))
    if feat_select == 'k_best':
        parameters['feat_select__k'] = tuple(range(1, n+1))
    if clf == 'svm':
        parameters['clf__C'] = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
        parameters['clf__gamma'] = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)
    if clf == 'ada_boost':
        parameters['clf__n_estimators'] = (100, 300, 500)
        parameters['clf__learning_rate'] = (0.5, 1.0, 1.5)
    return parameters

def _get_best_parameters(feat_select, clf):
    best_parameters = {}
    if feat_select == 'pca' and clf == 'svm':
        best_parameters['dim_reduct__n_components'] = 4
        best_parameters['clf__C'] = 1.0e5
        best_parameters['clf__gamma'] = 1.0e-2
        best_parameters['clf__kernel'] = 'rbf'
    if feat_select == 'k_best' and clf == 'svm':
        best_parameters['feat_select__k'] = 1#6
        best_parameters['clf__C'] = 75.0
        best_parameters['clf__gamma'] = 0.5
        best_parameters['clf__kernel'] = 'rbf'
    if feat_select == 'pca' and clf == 'ada_boost':
        best_parameters['dim_reduct__n_components'] = 3
        best_parameters['clf__n_estimators'] = 100
        best_parameters['clf__learning_rate'] = 1.0
    if feat_select == 'k_best' and clf == 'ada_boost':
        best_parameters['feat_select__k'] = 1#7
        best_parameters['clf__n_estimators'] = 500
        best_parameters['clf__learning_rate'] = 1.5
    if feat_select == 'pca' and clf == 'nb':
        best_parameters['dim_reduct__n_components'] = 4
    return best_parameters

def _get_new_features(pipeline):
    '''
    gets either the 'dim_reduct' or 'feat_select' step from the pipeline
    after optimization
    :param pipeline:
    :return: PCA components if step if 'dim_reduct',
            selected features if step is 'feat_select'
    '''
    step = pipeline.named_steps.get('feat_select')
    if step:
        ind_selected_feat = step.get_support(indices=True)
        print('Indices of selected features: {0}'.format(ind_selected_feat))
        myfeatures = np.array(_get_features()[1:])
        selected_feat = myfeatures[ind_selected_feat]
        print('selected features: {0}'.format(selected_feat))
        return np.insert(selected_feat, 0, 'poi')

def _get_new_classifier(pipeline):
    '''

    :param pipeline:
    :return: new classifier after the optimization
    '''
    clf = pipeline.named_steps.get('clf')
    print('clf: {0}'.format(clf))
    return clf


def _evaluate_grid_search(grid_search, mypipeline, parameters, feature, label, scoring):
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in mypipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(feature, label)
    print("done in {0:.3f} s".format(time() - t0))
    #print(grid_search.cv_results_)
    print("Scorer: {0}".format(grid_search.scorer_))
    print("Best score: {0:.3f}".format(grid_search.best_score_))
    print("Best estimator: {0}".format(grid_search.best_estimator_))
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def _get_pipeline_and_parameters(feat_select, clf, feat_select_object, clf_object, features, labels, num_features):
    if feat_select == 'pca' and clf == 'svm':
        mypipeline = Pipeline([('dim_reduct', feat_select_object), ('clf', clf_object)])
        parameters = _get_parameters(feat_select='pca', clf='svm', n=num_features)
        best_parameters = _get_best_parameters(feat_select='pca', clf='svm')
        mypipeline_with_params = mypipeline.set_params(**best_parameters)
        mypipeline_with_params.fit(features, labels)
    if feat_select == 'k_best' and clf == 'svm':
        mypipeline = Pipeline([('feat_select', feat_select_object),('clf', clf_object)])
        parameters = _get_parameters(feat_select='k_best', clf='svm', n=num_features)
        best_parameters = _get_best_parameters(feat_select='k_best', clf='svm')
        mypipeline_with_params = mypipeline.set_params(**best_parameters)
        mypipeline_with_params.fit(features, labels)
    if feat_select == 'pca' and clf == 'ada_boost':
        mypipeline = Pipeline([('dim_reduct', feat_select_object), ('clf', clf_object)])
        parameters = _get_parameters(feat_select='pca', clf='ada_boost', n=num_features)
        best_parameters = _get_best_parameters(feat_select='pca', clf='ada_boost')
        mypipeline_with_params = mypipeline.set_params(**best_parameters)
        mypipeline_with_params.fit(features, labels)
    if feat_select == 'k_best' and clf == 'ada_boost':
        mypipeline = Pipeline([('feat_select', feat_select_object), ('clf', clf_object)])
        parameters = _get_parameters(feat_select='k_best', clf='ada_boost', n=num_features)
        best_parameters = _get_best_parameters(feat_select='k_best', clf='ada_boost')
        mypipeline_with_params = mypipeline.set_params(**best_parameters)
        mypipeline_with_params.fit(features, labels)
    if feat_select == 'pca' and clf == 'nb':
        mypipeline = Pipeline([('dim_reduct', feat_select_object), ('clf', clf_object)])
        parameters = _get_parameters(feat_select='pca', clf='nb', n=num_features)
        best_parameters = _get_best_parameters(feat_select='pca', clf='nb')
        mypipeline_with_params = mypipeline.set_params(**best_parameters)
        mypipeline_with_params.fit(features, labels)
    return mypipeline, mypipeline_with_params, parameters, best_parameters

def _test_pipeline(pipeline, params, feature_train, label_train, data_dict, features_list, with_new_features, folds):
    """
    evaluates the classifier for all parameters using tester.py
    :param pipeline:
    :param params:
    :param feature_train:
    :param label_train:
    :param data_dict:
    :param features_list:
    :param folds: number of folds in StratifiedShuffleSplit
    :return: score_stats, {'precision': (mean, std), 'recall': (mean, std), 'accuracy': (mean, std), 'clf': clf }
    """
    params_names = params.keys()
    params_values = list(params.values())
    params_values_product = list(itertools.product(*params_values))
    score_stats_list = []
    for value_set in params_values_product:
        kwargs = {name: value for name, value in zip(params_names, value_set)}
        print('parameters: {0}'.format(kwargs))
        pipeline.set_params(**kwargs).fit(feature_train, label_train)
        score_stats = test_classifier(pipeline, data_dict, features_list, with_new_features, folds)
        if score_stats:
            score_stats_list.append(copy.deepcopy(score_stats))
        #print('score_stats_list: {0}'.format(score_stats_list))
    _find_best_params(score_stats_list)

def _find_best_params(score_stats_list):
    """
    input: [{'precision': (mean1, std1), 'recall': (mean1, std1), 'accuracy': (mean1, std1), 'clf': clf1 },
            {'precision': (mean2, std2), 'recall': (mean2, std2), 'accuracy': (mean2, std2), 'clf': clf2 },
            ...]
    :param score_stats_list:
    :return:
    """
    precision_list = []
    recall_list = []
    accuracy_list = []
    for element in score_stats_list:
        try:
            precision = element['precision'][0]
            recall = element['recall'][0]
            accuracy = element['accuracy'][0]
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
        except TypeError as err:
            print(err)
    for score_list, score in zip([precision_list, recall_list, accuracy_list], ['precision', 'recall', 'accuracy']):
        try:
            score_array = np.array(score_list)
            max_score = score_array.max()
            index_max_score = score_array.argmax()
            max_clf = score_stats_list[index_max_score]['clf']
            #print('all data for maximum {0}: {1}'.format(score, score_stats_list[index_max_score]))
            print('maximum {0}: {1:0.3f}'.format(score, max_score))
            print('clf for maximum {0}: {1}'.format(score, max_clf))
        except (TypeError, ValueError) as err:
            print(err)