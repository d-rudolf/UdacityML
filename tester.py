#!/usr/bin/pickle

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
#sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f} \tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, with_new_features, folds = 10):
    data = featureFormat(dataset, feature_list,
                         remove_NaN=True,
                         remove_all_zeroes=True,
                         remove_any_zeroes=False,
                         sort_keys=True)
    labels, features = targetFeatureSplit(data)
    if with_new_features:
        print('with new features')
        features = _create_new_features(features)
    features = preprocessing.scale(features)
    cv = StratifiedShuffleSplit(n_splits=folds, test_size=0.25, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    accuracy_list = []
    precision_list = []
    recall_list = []
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
        try:
            accuracy, recall, precision = _calculate_scores(true_positives, true_negatives, false_positives, false_negatives)
            accuracy_list.append(accuracy)
            recall_list.append(recall)
            precision_list.append(precision)
        except TypeError as err:
            pass
            #print(err)
    try:
        myscores = {'accuracy': accuracy_list, 'recall': recall_list, 'precision': precision_list}
        score_stats = {}
        if recall_list:
            mean_recall, std_recall = _get_mean_and_std(recall_list)
        else:
            mean_recall = 0.0
        if mean_recall > 0.25:
            for k, v in myscores.items():
                mean, std = _get_mean_and_std(v)
                print('{0}: {1:.3f} +/- {2:.3f}'.format(k, mean, std))
                score_stats[k] = (mean, std)
            print('clf: {0}'.format(clf))
            score_stats['clf'] = clf
            total_predictions = true_negatives + false_negatives + false_positives + true_positives
            accuracy = 1.0*(true_positives + true_negatives)/total_predictions
            precision = 1.0*true_positives/(true_positives+false_positives)
            recall = 1.0*true_positives/(true_positives+false_negatives)
            f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
            f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
            print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
            print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
            #print("score_stats: {0}".format(score_stats))
            return score_stats
    except ZeroDivisionError as err:
        print("Got a divide by zero when trying out: {0}".format(clf))
        print("Precision or recall may be undefined due to a lack of true positive predictions.")


def _calculate_scores(true_positives, true_negatives, false_positives, false_negatives):
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        #print('total predictions: {0:0.3f}'.format(total_predictions))
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        #print('total accuracy: {0:0.3f}'.format(accuracy))
        precision = 1.0 * true_positives / (true_positives + false_positives)
        #print('total precision: {0:0.3f}'.format(precision))
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        #print('total recall: {0:0.3f}'.format(recall))
        return accuracy, recall, precision
    except ZeroDivisionError as err:
        pass
        #print(err)

def _get_mean_and_std(list):
    mean = np.array(list).mean()
    std = np.array(list).std()
    return mean, std

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

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "bw") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "bw") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "bw") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "br") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "br") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "br") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    print('clf: {0}'.format(clf))
    print('feature list: {0}'.format(feature_list))
    ### Run testing script
    test_classifier(clf, dataset, feature_list, with_new_features=False)

if __name__ == '__main__':
    main()
