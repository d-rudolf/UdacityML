Review #1 (this review)
-----------------------

Reviewed 5 days ago

student notes

_None provided_

Review #1 (this review)
-----------------------

Reviewed 5 days ago

student notes

_None provided_

**helper.py** 3

**tester.py**

**report.md**

**poi_names.txt**

**poi_id.py**

**poi\_email\_addresses.py**

**flask_app/manage.py**

**flask_app/app/views.py**

**flask_app/app/static/js/popper.min.js**

**flask\_app/app/static/js/plot\_data.js**

**flask_app/app/static/js/jquery-3.2.1.min.js**

**flask_app/app/static/js/d3.min.js**

**flask_app/app/static/js/bootstrap.min.js**

**flask_app/app/static/js/README.md**

**flask_app/app/static/js/CHANGES.md**

**flask_app/app/static/js/API.md**

**flask_app/app/static/css/bootstrap.min.css**

**flask\_app/app/\_\_init__.py**

**README.md**

Your reviewer has provided annotations for your project

[Download annotations](https://review.udacity.com/)

Share your accomplishment

 ![](./Udacity Reviews_files/twitter.svg) ![](./Udacity Reviews_files/facebook.svg) 

Share your accomplishment! [![](./Udacity Reviews_files/twitter.svg)](https://review.udacity.com/) [![](./Udacity Reviews_files/facebook.svg)](https://review.udacity.com/)

### Requires Changes

#### 9 specifications require changes

This is a very good first attempt at the project! I left comments explaining what needs to be updated in both the code and report. Good luck and we look forward to reviewing the resubmission!

### Quality of Code

Code reflects the description in the answers to questions in the writeup. i.e. code performs the functions documented in the writeup and the writeup clearly specifies the final analysis strategy.

#### Required

This section can be evaluated once the report and code have been updated based on the feedback below.

poi_id.py can be run to export the dataset, list of features and algorithm, so that the final algorithm can be checked easily using tester.py.

#### Required

The `poi_id.py` file threw the following error:

    Traceback (most recent call last):
      File "poi_id.py", line 25, in <module>
        with open("final_project_dataset.pkl", "br") as data_file:
    ValueError: mode string must begin with one of 'r', 'w', 'a' or 'U', not 'br'
    
#### Answer
My code is written for the Python 3.5 interpreter. This error occurs with the Python 2.7 interpreter.
Please follow my instructions from README.md to run the code.


### Understanding the Dataset and Question

Student response addresses the most important characteristics of the dataset and uses these characteristics to inform their analysis. Important characteristics include:

*   total number of data points
*   allocation across classes (POI/non-POI)
*   number of features used
*   are there features with many missing values? etc.

#### Required

The report should at least include the following key characteristics:

1.  The number of data points
2.  The number of POIs

Student response identifies outlier(s) in the financial data, and explains how they are removed or otherwise handled.

#### Answer

The number of data points (persons) before formatting is 146 and after having removed all persons where all values are zero
there are 136 persons left.
The number of POI/non POI is 18/118.  

#### Required

Nice work in removing the `TOTAL` outlier. But outliers like `LAY KENNETH L` and `SKILLING JEFFREY K` should be retained in the data because there aren't many POIs.

#### Answer

I tried to build a pipeline keeping 'LAY KENNETH L' and 'SKILLING JEFFREY K' in the dataset but it performs worse than without them.
I can only fulfil the requirements recall and precision < 0.3 if I remove them.    
Also, removing 2 from 18 POI should not be a problem.
### Optimize Feature Selection/Engineering

At least one new feature is implemented. Justification for that feature is provided in the written response. The effect of that feature on final algorithm performance is tested or its strength is compared to other features in feature selection. The student is not required to include their new feature in their final feature set.

#### Required

In addition to creating the new features, please provide a response to explain the effect that the new features have on the final algorithm. This can be done by training a simple classifier with and without the new features. The feature importance scores of all the features (new and existing) could also be provided to show the strength of the new features.


#### Answer
I tested my pipeline with and without the new features. 
Here are the results without new features, i.e. with theses features:
['from_poi_to_this_person','to_messages','from_this_person_to_poi','from_messages','salary','bonus','deferral_payments','deferred_income','director_fees','expenses','loan_advances','long_term_incentive','total_payments'] 
maximum precision: 0.650

clf for maximum precision: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=11, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=20.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
  
maximum recall: 0.454

clf for maximum recall: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=80.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

maximum accuracy: 0.890

clf for maximum accuracy: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=11, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=20.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

The results with new features, i.e. ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio', 'salary','bonus','deferral_payments','deferred_income','director_fees','expenses','loan_advances','long_term_incentive','total_payments'] 
are the same. I get the same values for maximum recall, precision and accuracy for the same pipeline parameters.
From that I conclude that my new features are as good as the given ones and keep using them. 


Univariate or recursive feature selection is deployed, or features are selected by hand (different combinations of features are attempted, and the performance is documented for each one). Features that are selected are reported and the number of features selected is justified. For an algorithm that supports getting the feature importances (e.g. decision tree) or feature scores (e.g. SelectKBest), those are documented as well.

#### Required

Since the features were chosen manually, provide the performance scores (precision and recall) for the combinations of features that were tested. By doing this, we can select the combination that yielded the best results.

### Answer
I manually tried a different subset of features and obtained these results.

features: 

['from_poi_to_this_person_ratio','from_this_person_to_poi_ratio','salary','bonus','deferral_payments','deferred_income','director_fees','expenses','loan_advances','long_term_incentive','total_payments','exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value']

maximum precision: 0.448

clf for maximum precision: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=80.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

maximum recall: 0.187

clf for maximum recall: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=7, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=85.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

maximum accuracy: 0.886

clf for maximum accuracy: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=80.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

features:

['from_poi_to_this_person_ratio','from_this_person_to_poi_ratio','salary','bonus','deferral_payments','deferred_income','director_fees','expenses']

maximum precision: 0.713

clf for maximum precision: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=7, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

maximum recall: 

0.461
clf for maximum recall: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=75.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

maximum accuracy: 0.886

clf for maximum accuracy: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

features:

['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio', 'from_messages', 'salary', 'bonus', 'deferral_payments']

maximum precision: 0.395

clf for maximum precision: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

maximum recall: 0.248

clf for maximum recall: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=20.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

maximum accuracy: 0.852

clf for maximum accuracy:

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

The best result I obtained for these features:
['from_poi_to_this_person_ratio','from_this_person_to_poi_ratio','salary','bonus','deferral_payments','deferred_income','director_fees','expenses']



If algorithm calls for scaled features, feature scaling is deployed.

### Pick and Tune an Algorithm

At least two different algorithms are attempted and their performance is compared, with the best performing one used in the final analysis.

Response addresses what it means to perform parameter tuning and why it is important.

#### Awesome

The report thoroughly addresses parameter tuning and why it is important. Good job!

At least one important parameter tuned with at least 3 settings investigated systematically, or any of the following are true:

*   GridSearchCV used for parameter tuning
*   Several parameters tuned
*   Parameter tuning incorporated into algorithm selection (i.e. parameters tuned for more than one algorithm, and best algorithm-tune combination selected for final analysis).

#### Suggestion

In addition to testing several parameters, it would be good to include the various settings testing for each. In the report, it is recommended to include at least three settings for each parameter.

### Validate and Evaluate

At least two appropriate metrics are used to evaluate algorithm performance (e.g. precision and recall), and the student articulates what those metrics measure in context of the project task.

#### Required

When defining precision and recall, try to use more layman's terms instead of terms like 'True Positives' or 'False Positives' so that anyone can understand these metrics.

#### Answer

Thank you, I changed my explanation accordingly.

Response addresses what validation is and why it is important.

#### Required

When defining validation, explain why the data has to be partitioned. What would happen if the data wasn't partitioned and the model was trained using the entire dataset? And what is the main purpose of using the test set?

Here's a link that may help with defining validation: [https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_233](https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_233)

#### Answer

I changed my explanation accordingly.


Performance of the final algorithm selected is assessed by splitting the data into training and testing sets or through the use of cross validation, noting the specific type of validation performed.

#### Suggestion

In addition to mentioning that the Stratified Shuffle Split was used, it would be good to explain why it is the preferred validation method for the project task.

When tester.py is used to evaluate performance, precision and recall are both at least 0.3.

#### Required

The final algorithm couldn't be evaluated because the `poi_id.py` file should be able to generate new pickle files. This is because there are times when the provided pickle files will throw errors, like the one below:

    Traceback (most recent call last):
      File "tester.py", line 103, in <module>
        main()
      File "tester.py", line 98, in main
        clf, dataset, feature_list = load_classifier_and_data()
      File "tester.py", line 91, in load_classifier_and_data
        clf = pickle.load(open(CLF_PICKLE_FILENAME, "r") )
      File "lib\pickle.py", line 1384, in load
        return Unpickler(file).load()
      File "lib\pickle.py", line 864, in load
        dispatch[key](self)
      File "lib\pickle.py", line 892, in load_proto
        raise ValueError, "unsupported pickle protocol: %d" % proto
    ValueError: unsupported pickle protocol: 3
    
#### Answer:

Please use Python 3.5 to read my pickle files.