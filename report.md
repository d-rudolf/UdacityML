## Udacity project 'Fraud Identification from Enron Emails'

#### Note

Please use the Python 3.5 interpreter to run the code and load the pickle files.
The code is not compatible with Python 2.7.
Please follow the instructions in README.md for initial setup.

#### 1. Goal of this project and how machine learning is useful in trying to accomplish it (data exploration, outlier investigation)

The data for this project is partly from the public Enron email data set (https://www.cs.cmu.edu/~enron/) and partly 
from financial data from findlaw.com (http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf). 
The data can be roughly classified in three classes, which are income and stock data and email statistics. 
There is also a feature shared receipt with poi, which does not fall into any of the three classes. 
The features are summarized in Tab. 1.  
<table>  
    <tr>
        <td>income data </td> 
        <td> stock data </td>  
        <td> email statistics </td> 
        <td> misc </td> </tr>
    <tr>
        <td>salary, bonus, deferral payments, deferred income, 
        director fees, expenses, loan advances, long term incentive,
        total payments
        </td> 
        <td> exercised stock options, restricted stock, 
        restricted stock deferred, total stock value 
        </td>
        <td>number of total from/to messages, number of messages
        from/to 
        poi
        </td>
        <td> shared receipt with poi 
     <tr>
</table>
Tab. 1

The goal is to build a classifier based on the features in Tab. 1 that correctly predicts 
whether a person is involved in fraud or not (poi = 1 or poi = 0).
To visualize the data I wrote a small web app, which you can run from the flask_app folder:
```
mypath/UdacityML/flask_app$ python manage.py runserver
```  
The number of data points (persons) before formatting is 146 and after having removed all persons where all values are zero
there are 136 persons left.
The number of POI/non POI is 18/118.  
If you define outliers as data points, which are several standard deviations away from the mean of a distribution, 
I found three outliers:  'TOTAL', 'LAY KENNETH L', 'SKILLING JEFFREY K'. I tried to build a pipeline keeping 
'LAY KENNETH L' and 'SKILLING JEFFREY K' in the dataset but it performs worse than without them.
I can only fulfil the requirements recall and precision < 0.3 if I remove them.    

#### 2. Features (create new features, intelligently select features, properly scale features)
First, I selected the financial features:
<br> 
['salary', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'expenses', 'loan_advances', 
'long_term_incentive', 'total_payments']. In addition, I created two new features:
<br> 
'from_poi_to_this_person_ratio' ='from_poi_to_this_person'/'to_messages' and 
'from_this_person_to_poi_ratio'='from_this_person_to_poi'/'from_messages', i.e. a ratio between  
from_poi and to_poi and the total in and out emails. This reduces the number of features from four to two 
and creates new reasonable features. 
I tested my pipeline with and without the new features. 
Here are the results without new features, i.e. with these features:

['from_poi_to_this_person','to_messages','from_this_person_to_poi','from_messages','salary','bonus','deferral_payments','deferred_income','director_fees','expenses','loan_advances','long_term_incentive','total_payments'] 

#####maximum precision: 0.650

#####clf for maximum precision: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=11, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=20.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
  
##### maximum recall: 0.454

#####clf for maximum recall: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=80.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

#####maximum accuracy: 0.890

#####clf for maximum accuracy: 

Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=11, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=20.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])

The results with new features, i.e. 

['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio', 'salary','bonus','deferral_payments','deferred_income','director_fees','expenses','loan_advances','long_term_incentive','total_payments'] 

are the same. I get the same values for maximum recall, precision and accuracy for the same pipeline parameters.
From that I conclude that my new features are as good as the given ones and keep using them. 
Using these features I can achieve a precision and recall higher than 0.3 (Tab. 2).       
I selected these features by testing a different subset of features with the tester.py, which creates a  
StratifiedShuffleSplit cross validator and calculates the total accuracy, precision and recall.
Initially, I also tried GridSearchCV to find the best model parameters, but it gave no agreement with the results 
from tester.py when defining recall or precision as scores. Therefore, I wrote a function that iterates over the 
parameters like n_components of PCA or C and gamma of SVM and selects the the parameters for the highest accuracy, 
precision and recall.
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

At this point, I also have to make a criticism that originally, in tester.py, precision, recall 
and accuracy are calculated as some kind of global values for all folds of StratifiedShuffleSplit. I think, a better 
approach is to have a distribution of precision, recall and accuracy values, where each value stems from a single fold.
Then, it is possible to calculate a mean and a standard deviation, which measures the quality of the prediction.        
I modified tester.py accordingly to obtain these values for each fold.   
For scaling, I used the preprocessing.scale, which standardizes the data set, because algorithms like SVM expect a 
standardized data set.   


#### 3. Algorithm (pick an algorithm)
<table>  
    <tr>
        <td> features </td> 
        <td> feature weights </td>  
        <td> best pipeline </td> 
        <td> accuracy </td> 
        <td> precision </td> 
        <td> recall </td> 
    </tr>
    <tr>
        <td> 'from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio',
             'salary', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'expenses'</td> 
        <td>  </td>  
        <td> Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=75.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
         </td> 
        <td> 0.869 +/- 0.011</td> 
        <td> 0.461 +/- 0.040 </td> 
        <td> 0.461 +/- 0.124 </td> 
     <tr>
     <tr>
        <td>  </td> 
        <td>  </td>  
        <td>  Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=7, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=100, random_state=None))])       </td> 
        <td> 0.890 +/- 0.009 </td> 
        <td> 0.570 +/- 0.052 </td> 
        <td> 0.395 +/- 0.042 </td> 
     <tr>
     <tr>
        <td>  </td> 
        <td>[  0.19415698   6.11394231   6.42033595   2.73336044   0.7804664
            15.80286711   1.37297104   4.1088099 ] 
        </td>  
        <td>  Pipeline(memory=None,
     steps=[('feat_select', SelectKBest(k=6, score_func=<function f_classif at 0x7f6206477ae8>)), ('clf', SVC(C=75.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])  
        </td> 
        <td> 0.850 +/- 0.016 </td> 
        <td> 0.369 +/- 0.043 </td> 
        <td> 0.312 +/- 0.075</td> 
     <tr>
     <tr>
        <td>  </td> 
        <td> [  0.19415698   6.11394231   6.42033595   2.73336044   0.7804664
         15.80286711   1.37297104   4.1088099 ]
        </td>  
        <td> Pipeline(memory=None,
     steps=[('feat_select', SelectKBest(k=7, score_func=<function f_classif at 0x7f03045b2ae8>)), ('clf', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.5, n_estimators=500, random_state=None))])
        </td> 
        <td> 0.828 +/- 0.010 </td> 
        <td> 0.307 +/- 0.038 </td> 
        <td> 0.338 +/- 0.061 </td> 
     <tr>
</table>
Tab. 2
<br>
I tried KBest and PCA for feature selection and dimensionality reduction and SVM and AdaBoost for classification.
The results are summarized in Tab. 2, where you can see the features, the feature weights, my best pipeline with all 
the parameters and the obtained scores (accuracy, precision and recall). My best result was with PCA and SVM 
(row 1 in Tab. 2). The difficult part was to get both recall and precision above the threshold of 0.3.

#### 4. Parameter tuning
Tuning the parameters of an algorithm means trying to maximize a certain score (e.g. precision and/or recall) varying 
the parameters, e.g. C and gamma of SVM or k in SelectKBest.
If the parameters are not well tuned the generalization behaviour can be unsatisfactory, i.e. high score on the 
training but low score on the test data set (overfitting). Or you can just have a bad model with a low score even on the
training data set (underfitting). I tuned the parameters of SelectKBest (k), PCA (n_components), SVM (C, gamma) and 
AdaBoost (n_estimators, learning_rate) with my own testing routine (_test_pipeline in poi_id.py), which you will find 
in the code. It basically scans a given parameter range and calculates the accuracy, precision and recall for each 
parameter set using the functions in helper.py and tester.py.    

#### 5. Valdidation strategy
Validation implies that the data set is split in training and testing data. Validation itself is applying 
a classification or regression, which was fit to the training data, on the testing data. 
If the model would be build on the entire data set without partitioning it, the model might not generalize well, i.e. 
it can potentially score badly on new data.
For cross validation I used StratifiedShuffleSplit in tester.py. 

#### 6. Usage of evaluation metrics
The evaluation metrics in tester.py are precision and recall. Their definition is:
<br>
precision = number of true positives/(number of false positives + number of true positives)
<br>
recall = number of true positives/(number of false negatives + number of true positives)
My results for both are presented in Tab. 2. 
For my data set, a precision of 0.349 means e.g. that there are 15 true positives and 28 false positives. 
A recall of 0.375 means that there are 15 true positives and 25 false negatives.
If a person is a POI, a classifier with a high precision will most likely correctly classify him/her as a POI.     
If a person is not a POI, a classifier with a high recall will most likely correctly classify him/her as a non-POI.     
 

#### 7. References
- documentation of scikit-learn (http://scikit-learn.org/stable/) 
- Udacity course on 'Intro to Machine Learning'

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that 
were taken from websites, books, forums, blog posts, github repositories, etc..
