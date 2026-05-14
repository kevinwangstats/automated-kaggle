import pandas as pd
from sklearn.datasets import make_classification
import h2o
from h2o.sklearn import H2OAutoMLClassifier

X, y = make_classification(n_samples=100, random_state=42)
h2o.init()
clf = H2OAutoMLClassifier(max_models=3, seed=42)
clf.fit(X, y)
print(clf.predict(X))
