import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import h2o
from h2o.sklearn import H2OAutoMLClassifier

df = pd.read_csv("data/titanic/train.csv")
X = df.drop(columns=["Survived", "Name", "Ticket", "Cabin"])
y = df["Survived"]

cat_cols = X.select_dtypes(include=['object', 'category']).columns
num_cols = X.select_dtypes(include=np.number).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)

h2o.init(verbose=False)
H2OAutoMLClassifier._estimator_type = "classifier"
clf = H2OAutoMLClassifier(max_models=1, seed=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

try:
    pipeline.fit(X, y)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
