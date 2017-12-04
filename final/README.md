## CEBD 1260 Final Programming Assignment

Complete both questions in a single jupyter notebook and upload to github repo in a new directory named "final".

Answers should include both code and written explanation / interpretation of results. Be sure to answer all parts of the question completely.

1) Your first task to is to classify data from a cancer diagnostic database. In this database are patients with tumors, characteristics of those tumors, and biospy results indicating whether the tumor is Malignant or Benign.

In cancer_data.txt you will find the following variables:

   - radius (mean of distances from center to points on the perimeter)
   - texture (standard deviation of gray-scale values)
   - perimeter
   - area
   - smoothness (local variation in radius lengths)
   - compactness (perimeter^2 / area - 1.0)
   - concavity (severity of concave portions of the contour)
   - concave_points (number of concave portions of the contour)
   - symmetry 
   - fractal_dimension ("coastline approximation" - 1)
   - cancer (0 = Benign, 1 = Malignant)  *target*


Use any machine learning algorithm you wish. In your answer include a short description of your algorithm of choice and predicted category of a new patient with a tumor with the following features:

   - radius: 14
   - texture: 14
   - perimeter: 88
   - area: 566
   - smoothness: 1
   - compactness: 0.08
   - concavity: 0.06
   - concae points: 0.04
   - symmetry: 0.18
   - fractal dimension: 0.05
2) The following code contains a 5 bugs (errors). Find and correct them all and then answer the following questions:

  1. How many observations are in the training dataset?
  2. How many features are in the training dataset?
  3. How well did your model perform?

  BONUS: Which category is Hockey? 0 or 1? Which category is baseball?
  
  ```
 from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import pipeline
from sklearn_linear_model import SGDClassifier

categories = [ 'rec.sport.baseball','rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='test', categories=categories)
twenty_test = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42)),
])
text_clf.fit(twenty_train.data, twenty_test.target)  
predicted = text_clf.predict(twenty_test.data)
  ```
