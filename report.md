## Udacity project 'Fraud Identification from Enron Emails'

#### Note

Please use the Python 3.5 interpreter to run the code and load the pickle files.
The code is not compatible with Python 2.7.
Please follow the instructions in README.md for initial setup.
The project report is in the file report.md.  
You can test my project using the original file tester.py but you have to make it compatible with Python 3.5.
Initially, it is not compatible with Python 3.5. Also, since I did feature scaling with preprocessing.scale(features) 
you also have to include this line of code into the original tester.py
```
features = preprocessing.scale(features)

```
Otherwise the features will be not scaled and my classifier will not work (makes sense, right?).  
Alternatively, you can also run my version of tester.py to test the classifier, which incorporates feature scaling.
Checking the code you will see what I did.
This is my fourth submission and I tried really hard to fulfill all the requirements. I hope I can pass the project now.

#### 1. Goal of this project and how machine learning is useful in trying to accomplish it (data exploration, outlier investigation)

The data for this project is partly from the public Enron email data set (https://www.cs.cmu.edu/~enron/) and partly 
from financial data from findlaw.com (http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf). 
The data can be roughly classified in three classes, which are income and stock data and email statistics. 
There is also a feature 'shared receipt with poi', which does not fall into any of the three classes. 
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
I found three outliers:  'TOTAL', 'LAY KENNETH L', 'SKILLING JEFFREY K'. 
According to the feedback from the previous review I removed only 'TOTAL' and kept 'LAY KENNETH L' and 'SKILLING JEFFREY K'
in the data set.

#### 2. Features (create new features, intelligently select features, properly scale features)

My procedure to select the best features was as follows. First, I ordered all features by their SelectKBest score (ANOVA F-value):
[('exercised_stock_options', 34.73257743), ('total_stock_value', 33.31087215), ('restricted_stock', 15.21368276), ('salary', 10.22957239), ('bonus', 10.09863783), ('total_payments', 8.82481433), ('loan_advances', 8.17870326), ('long_term_incentive', 7.54929549), ('other', 4.85790445), ('expenses', 3.34279872), ('from_this_person_to_poi', 2.16001363), ('shared_receipt_with_poi', 1.38336448), ('director_fees', 1.37250905), ('deferred_income', 1.19555209), ('deferral_payments', 0.8831138), ('from_poi_to_this_person', 0.75033658), ('to_messages', 0.57357048), ('restricted_stock_deferred', 0.07779663), ('from_messages', 0.05916248)]
Then, I selected the first, the first and the second, the first, the second and the third etc. features and performed a 
parameter scan with my own function _test_pipeline from helper.py. The function iterates over the parameters like n_components of PCA or C and gamma of SVM and selects the parameters for 
the highest accuracy, precision and recall. 
I did not use GridSearchCV because it did not worked for 
me defining precision and recall as scoring. A log file from the parameter scan is attached (parameter_scan_pca_svm.log).   
The best result I obtained for these features:
['exercised_stock_options', 'total_stock_value', 'restricted_stock', 'salary', 'bonus']
Their SelectKBest scores are [ 21.43451501  20.77586138   7.61960042  14.86411702  17.59161819].
The best result is 

precision: 0.507 +/- 0.032

recall: 0.478 +/- 0.072

accuracy: 0.849 +/- 0.012

clf: Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=4, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
	Accuracy: 0.85455	Precision: 0.52500	Recall: 0.42000 	F1: 0.46667	F2: 0.43750
	Total predictions:  330	True positives:   21	False positives:   19	False negatives:   29	True negatives:  261

I also created a new feature multiplying 'exercised_stock_options' with 'total_stock_value', i.e. my feature list is
['exercised_stock_options' x 'total_stock_value', 'total_stock_value', 'restricted_stock', 'salary', 'bonus'] with
[ 21.49624452  20.77586138   7.61960042  14.86411702  17.59161819] as their corresponding SelectKBest scores.
My new feature has almost the same score as 'exercised_stock_options'. However, the classification result is worse than before:

recall: 0.328 +/- 0.054

accuracy: 0.822 +/- 0.022

precision: 0.399 +/- 0.070

clf: Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=4, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
	Accuracy: 0.83636	Precision: 0.44737	Recall: 0.34000 	F1: 0.38636	F2: 0.35714
	Total predictions:  330	True positives:   17	False positives:   21	False negatives:   33	True negatives:  259

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
        <td> ['exercised_stock_options', 'total_stock_value', 'restricted_stock', 'salary', 'bonus']</td> 
        <td> [ 21.43451501  20.77586138   7.61960042  14.86411702  17.59161819] </td>  
        <td> Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=4, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
         </td> 
        <td> 0.849 +/- 0.012</td> 
        <td>  0.507 +/- 0.032 </td> 
        <td> 0.478 +/- 0.072 </td> 
     <tr>
     <tr>
        <td>  </td> 
        <td>  </td>  
        <td>  Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=100, random_state=None))])
</td> 
        <td> 0.796 +/- 0.058 </td> 
        <td> 0.352 +/- 0.098 </td> 
        <td> 0.317 +/- 0.054 </td> 
     <tr>
     <tr>
        <td>  </td> 
        <td>  </td>  
        <td>  Pipeline(memory=None,
     steps=[('dim_reduct', PCA(copy=True, iterated_power='auto', n_components=4, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', GaussianNB(priors=None))])  
        </td> 
        <td> 0.859 +/- 0.017 </td> 
        <td> 0.569 +/- 0.110 </td> 
        <td> 0.311 +/- 0.048</td> 
     <tr>
</table>
Tab. 2
<br>
I tried PCA for dimensionality reduction and SVM, AdaBoost and GaussianNB for classification.
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
If a person is a non-POI, a classifier with a high precision will most likely correctly classify him/her as a non-POI 
(number of false positives is low). It will also be successful in identifying POIs. 
If a person is a POI, a classifier with a high recall will most likely correctly 
classify him/her as a POI (number of false negatives is low).

#### 7. References
- documentation of scikit-learn (http://scikit-learn.org/stable/) 
- Udacity course on 'Intro to Machine Learning'

I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that 
were taken from websites, books, forums, blog posts, github repositories, etc..
