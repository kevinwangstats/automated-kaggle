---
name: feature-engineering
description: Machine learning feature engineering guidance. Focus on data representation, preprocessing, target encoding, interaction terms, missing value imputation, and scikit-learn pipeline transformations. Ensure compatibility with the Agentic AutoML tabular pipeline.
version: 1.0.0
---

# Feature Engineering Guidelines

## Overview

This skill provides comprehensive guidance for performing feature engineering and data preprocessing within the Agentic AutoML pipeline. The goal is to maximize cross-validation scores by improving the data representation while adhering strictly to pipeline constraints.

## 1. Automated Kaggle Pipeline Constraints (CRITICAL)

When proposing feature engineering code, you MUST adhere to the following architecture constraints:
* **No Hardcoded Paths:** Read `dataset_path`, `target_col`, `test_path` from the configuration via `load_config()`.
* **Pipeline Integrity:** Feature engineering MUST be implemented as part of a scikit-learn `Pipeline` or `ColumnTransformer`. This prevents data leakage during cross-validation.
* **ID Column Preservation:** The first column in the test set is typically the ID column. Capture it before dropping it from the feature set.
* **Sparse Matrix Quirks:** For scikit-learn 1.3+, `OneHotEncoder` requires `sparse_output=False` and `MissingIndicator` requires `sparse=False`.

## 2. Feature Creation Strategies

* **Polynomial & Interaction Terms:** Use `PolynomialFeatures` to capture non-linear relationships.
* **Binning:** Use `KBinsDiscretizer` to convert continuous variables into categorical bins, which helps tree-based models capture non-linear distributions.
* **Domain-Specific Logic:** Use `FunctionTransformer` to apply custom pandas operations (e.g., date parsing, string length extraction) within the pipeline. 
* **Pandas String Safety:** When engineering features on string/categorical columns, ALWAYS cast to string before applying operations to prevent `TypeError` on `NaN` floats (e.g., `df[col].astype(str).str.contains(r'\d')`).

## 3. Feature Selection & Pruning

Adding too many features can lead to overfitting. Actively prune uninformative features:
* **SelectKBest**: Keep only the top-performing features based on statistical tests (`f_classif`, `mutual_info_classif`, or regression equivalents).
* **VarianceThreshold**: Remove features with zero or near-zero variance.
* **SelectFromModel**: Use a secondary estimator (like `RandomForestClassifier` or `Lasso`) to select features based on importance weights.

## 4. Handling Unknowns and Missing Values

* **Categorical Imputation:** Use `SimpleImputer(strategy='most_frequent')` or `strategy='constant', fill_value='missing'`.
* **Numerical Imputation:** Use `SimpleImputer(strategy='median')` or `KNNImputer`.
* **Robust Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` or `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` must be used for all categorical variables to gracefully handle unseen categories in the test set.

## 5. High-Cardinality Categorical Safety
Standard `OneHotEncoder` will bloat the dataset and cause timeouts or memory limits if applied to columns with many unique values (e.g., names, IDs, high-cardinality categories).
* **Preferred Approach:** Use scikit-learn's `TargetEncoder` natively within the `ColumnTransformer` for high-cardinality features. 
* **Implementation Example:** 
```python
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('target_encode', TargetEncoder(target_type='continuous', smooth=10.0))
])
```
Strict Rule: Never attempt to manually map string categories using Python dictionaries (.map({})) unless there are fewer than 5 explicit, known categories.

## Example: Advanced Feature Engineering Pipeline

```python
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from utils import load_config

# 1. Configuration & Data Load
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--output_dir", type=str, default=".")
args = parser.parse_args()

config = load_config(args.config)
# df = pd.read_csv(...) # Read using config

# 2. Pipeline Definition
numeric_features = ['age', 'income', 'balance']
categorical_features = ['occupation', 'city']

# Custom transformer example
def extract_string_length(df):
    return df.astype(str).apply(lambda x: x.str.len())

string_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
    ('length', FunctionTransformer(extract_string_length, validate=False)),
    ('scaler', StandardScaler())
])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('str', string_transformer, ['notes'])
])

# Feature Selection and Modeling
model = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_classif, k=20)),
    ('classifier', LGBMClassifier(n_jobs=-1, verbosity=-1, random_state=42))
])

# 3. Fit and Predict
# model.fit(X_train, y_train)
```
