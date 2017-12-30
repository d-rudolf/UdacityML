### Requires Changes

#### 3 specifications require changes

Good job incorporating the changes previously recommended. There are a couple more important updates that are still required to pass all the specifications in this project. For more help, you may also try asking your assigned mentor for some assistance with this project.

### Quality of Code

Code reflects the description in the answers to questions in the writeup. i.e. code performs the functions documented in the writeup and the writeup clearly specifies the final analysis strategy.

poi_id.py can be run to export the dataset, list of features and algorithm, so that the final algorithm can be checked easily using tester.py.

I was able to run `poi_id.py` with python 3.5 as you have suggested.

****### Understanding the Dataset and Question

Student response addresses the most important characteristics of the dataset and uses these characteristics to inform their analysis. Important characteristics include:

*   total number of data points
*   allocation across classes (POI/non-POI)
*   number of features used
*   are there features with many missing values? etc.

All required characteristics were included in the report. Good work. These numbers are important in the analysis for following reasons:

1.  As you analyzed the allocation across classes, you will see that the data is unbalanced. This means cross-validation method like Stratified Shuffle Split is important since it makes sure the ratio of POI and non-POI is the same during training and testing.
2.  The imbalance data is also the reason why accuracy is not a good evaluation metric compared to, say, precision and recall.
3.  Number of data is relatively small, which means Stratified Shuffle Split combined with Grid Search CV is possible to use here with acceptable training time (see that Stratified Shuffle Split is also used in tester.py). For a larger dataset, you may want to look at methods such as Randomized Search CV.
4.  Report on NULL values is useful in deciding how to treat them i.e. whether to convert them to mean of the values or 0, or other fancy transformations.

Optionally, it might be a good idea to include these discussions in this section to demonstrate your understanding.

Student response identifies outlier(s) in the financial data, and explains how they are removed or otherwise handled.

Good job finding and removing TOTAL observation, a prominent outlier in the dataset. However, as previously mentioned, both LAY KENNETH L and SKILLING JEFFREY K should not be removed from test set since this is an important POI, and we have only a few POIs to begin with. The exception to this is if this observation is only removed from the training set. This is okay if sufficient justification was provided. For example, an outlier may be detrimental to the fit of a model due to its extreme value, but at the same time may be desirable to identify accurately as part of a test set.

### Answer

Yes, I kept LAY KENNETH L and SKILLING JEFFREY K in the data set. Therefore I also redid the whole analysis.

### Optimize Feature Selection/Engineering

At least one new feature is implemented. Justification for that feature is provided in the written response. The effect of that feature on final algorithm performance is tested or its strength is compared to other features in feature selection. The student is not required to include their new feature in their final feature set.

Good job comparing the performance with and without new features.

Univariate or recursive feature selection is deployed, or features are selected by hand (different combinations of features are attempted, and the performance is documented for each one). Features that are selected are reported and the number of features selected is justified. For an algorithm that supports getting the feature importances (e.g. decision tree) or feature scores (e.g. SelectKBest), those are documented as well.

I can see that various feature combinations have been tested, although it is a bit hard to read the report. I suggest writing the report in a more readable way e.g. by using a table.

(Optional) Iterative feature selection
--------------------------------------

One recommended way to choose the optimal number of features is by doing it iteratively. With this method, we run feature selection several times and measure the performance of the final model with these different number of selected features. We then select the number of features that corresponds to the highest model performance. The diagram below may better illustrate this concept; it uses F1-score as a way to measure the model's performance, but you may also use precision and/or recall.

[![feature_selection](./Udacity Reviews_2_files/feature_selection.png)](./Udacity Reviews_2_files/feature_selection.png)

If algorithm calls for scaled features, feature scaling is deployed.

### Pick and Tune an Algorithm

At least two different algorithms are attempted and their performance is compared, with the best performing one used in the final analysis.

Response addresses what it means to perform parameter tuning and why it is important.

At least one important parameter tuned with at least 3 settings investigated systematically, or any of the following are true:

*   GridSearchCV used for parameter tuning
*   Several parameters tuned
*   Parameter tuning incorporated into algorithm selection (i.e. parameters tuned for more than one algorithm, and best algorithm-tune combination selected for final analysis).

### Validate and Evaluate

At least two appropriate metrics are used to evaluate algorithm performance (e.g. precision and recall), and the student articulates what those metrics measure in context of the project task.

> If a person is a POI, a classifier with a high precision will most likely correctly classify him/her as a POI. If a person is not a POI, a classifier with a high recall will most likely correctly classify him/her as a non-POI.

This explanation of precision and recall is not correct. Aside from [wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall) page, I found that [this article](http://rushdishams.blogspot.co.id/2011/03/precision-and-recall.html) explains precision and recall in a way that leaves no ambiguity between the two.
### Answer
Thank you. The right statement reads:
 
If a person is a non-POI, a classifier with a high precision will most likely correctly classify him/her as a non-POI 
(number of false positives is low). It will also be successful in identifying POIs. 
If a person is a POI, a classifier with a high recall will most likely correctly 
classify him/her as a POI (number of false negatives is low).



Response addresses what validation is and why it is important.

Correct, the model would not generalize well, or in other words, overfit to training data.

In regards to cross-validation, one tip I can add here is to add another set when the dataset allows (i.e. larger dataset and more even distribution of classes). By dividing only into two sets (training and test data), it is possible that the classifier will "remember" the test data, as explained in this [Udacity's deep learning course](https://www.udacity.com/course/viewer#!/c-ud730/l-6370362152/m-6379811830). One solution to this is to create another separate dataset to test the final algorithm before publishing the classifier into the real-world (or submitting it for evaluation in machine learning competitions).

Performance of the final algorithm selected is assessed by splitting the data into training and testing sets or through the use of cross validation, noting the specific type of validation performed.

When tester.py is used to evaluate performance, precision and recall are both at least 0.3.

I was able to run the `poi_id.py` code properly, but running `tester.py` with produced pkl files got me the following result:

    Got a divide by zero when trying out: Pipeline(memory=None,
         steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=75.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])
    

To pass this specification, you need to ensure that `tester.py` code can be run with produced pkl files and precision and recall scores are both higher than 0.3. Here are the steps you can take to pass this requirement:

1.  Run `test_classifier()` function in `tester.py` with exported pkl files.
2.  Improve your feature selection process as recommended above i.e. using iterative feature selection.
3.  Include more values in GridSearchCV's parameters.

#### Answer

I cannot reproduce this error with my tester.py. I guess, you used the original version of tester.py. I modified it because
e.g. I use preprocessing.scale for feature scaling, which is not implemented in the original version. Please use my 
version of tester.py. 
   