import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/titanic/train.csv').dropna(subset=['Survived'])
X = df.drop(columns=['Survived'])
categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns
numerical_features = X.select_dtypes(include=np.number).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

out = preprocessor.fit_transform(X)
print(f"out type: {type(out)}, dtype: {out.dtype}")
