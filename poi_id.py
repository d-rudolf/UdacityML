#!/usr/bin/python
import sys
import pickle
from pprint import pprint
from time import time

from sklearn import svm, linear_model, preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from helper import _get_num_of_poi, _order_features_by_score, _create_new_features, _remove_outlier, _scale_data, _get_classifier, _cross_validate, _get_train_test_data, \
    _select_features, _get_parameters, _evaluate_grid_search, _get_features, _get_new_features, _get_new_classifier, \
    _test_pipeline, _get_best_parameters, _get_pipeline_and_parameters, _check_k_best_scores

sys.path.append("../tools/")

from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

def format_select_classify_test(features_list, test_pipeline=True, with_new_features = False):
    # len(features_list)-1 because of 'poi'
    num_features = len(features_list)-1
    data = featureFormat(data_dict_no_outlier, features_list,
                         remove_NaN=True,
                         remove_all_zeroes=True,
                         remove_any_zeroes=False,
                         sort_keys=True)
    """
    labels:  1 for poi, 0 for non-poi, features: np.array([])
    """
    labels, features = targetFeatureSplit(data)
    print('number of person after formatting: {0}'.format(len(features)))
    print('number of POI after formatting: {0}'.format(labels.count(1)))
    print('number of features: {0}'.format(len(features[0])))
    _check_k_best_scores(features, labels)
    if with_new_features:
        features = _create_new_features(features)
        _check_k_best_scores(features, labels)
    features = preprocessing.scale(features)
    feature_train, feature_test, label_train, label_test = _get_train_test_data(features, labels)
    """
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    """
    """
    possible values for feat_select: 'pca', 'k_best'
    """
    feat_select = 'pca'
    """
    possible values for clf: 'svm', 'ada_boost', 'nb'
    """
    clf = 'svm'
    feat_select_object = _select_features(feat_select)
    clf_object = _get_classifier(clf)
    mypipeline, mypipeline_with_params, parameters, best_parameters = \
        _get_pipeline_and_parameters(feat_select, clf, feat_select_object, clf_object, feature_train, label_train, num_features)
    select_k_best = mypipeline_with_params.named_steps.get('feat_select')
    if select_k_best:
        k_best_scores = select_k_best.scores_
        print('Feature scores in SelectKBest: {0}'.format(k_best_scores))
    """
    Grid Search
    GridSearchCV does not deliver results that I trust
    """
    scoring = ['accuracy', 'precision', 'recall']
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    grid_search = GridSearchCV(mypipeline, parameters, scoring=scoring,
                                                       cv = cv,
                                                       refit='recall',
                                                       verbose=0)
    #_evaluate_grid_search(grid_search, mypipeline, parameters, feature_train, label_train, scoring)

    """
    Task 5: Tune your classifier to achieve better than .3 precision and recall
    using our testing script. Check the tester.py script in the final project
    folder for details on the evaluation method, especially the test_classifier
    function. Because of the small size of the dataset, the script uses
    stratified shuffle split cross validation. For more info:
    http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    """
    test_classifier(mypipeline_with_params, data_dict_no_outlier, features_list, with_new_features, folds=10)
    if test_pipeline:
        _test_pipeline(mypipeline, parameters, feature_train, label_train, data_dict_no_outlier, features_list,
            with_new_features, folds=10)

    """
    Task 6: Dump your classifier, dataset, and features_list so anyone can
    check your results. You do not need to change anything below, but make sure
    that the version of poi_id.py that you submit can be run on its own and
    generates the necessary .pkl files for validating your results.
    """
    dump_classifier_and_data(mypipeline_with_params, data_dict_no_outlier, features_list)
"""
Task 1: Select what features you'll use.
features_list is a list of strings, each of which is a feature name.
The first feature must be "poi".
"""
features_list, score_list = _order_features_by_score()
with open("final_project_dataset.pkl", "br") as data_file:
    data_dict = pickle.load(data_file)
print('number of person before formatting: {0}'.format(len(data_dict.keys())))
num_poi = _get_num_of_poi(data_dict.values())
print('number of POI before formatting: {0}'.format(num_poi))
"""
### Task 2: Remove outliers
"""
data_dict_no_outlier = _remove_outlier(data_dict)
"""
### Task 3: Create new feature(s)
### Extract features and labels from dataset for local testing
"""
num_features = len(features_list)-1

#for i in range(2, num_features):
#    """
#    iterates through the features list selecting more and more features
#    features list is ordered by ANOVA F value from SelectKBest
#    feature_list[0]='poi'
#    """
#    myfeatures = features_list[0:i]
#    format_select_classify_test(myfeatures)

myfeatures = features_list[0:6]
print(myfeatures)
format_select_classify_test(myfeatures, test_pipeline=False, with_new_features=False)