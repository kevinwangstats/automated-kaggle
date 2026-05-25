import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/titanic/train.csv').dropna(subset=['Survived'])
X = df.drop(columns=['Survived'])
try:
    categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns
except TypeError:
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print(f"Total columns: {X.columns.tolist()}")
print(f"Categorical features: {categorical_features.tolist()}")
print(f"Numerical features: {numerical_features.tolist()}")
remaining_cols = set(X.columns) - set(categorical_features) - set(numerical_features)
print(f"Remaining cols: {remaining_cols}")

