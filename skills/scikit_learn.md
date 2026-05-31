---
name: scikit-learn
description: Machine learning in Python with scikit-learn. Use when working with supervised learning (classification, regression), unsupervised learning (clustering, dimensionality reduction), model evaluation, hyperparameter tuning, preprocessing, or building ML pipelines. Provides comprehensive reference documentation for algorithms, preprocessing techniques, pipelines, and best practices. Includes project-specific Agentic AutoML guardrails.
license: BSD-3-Clause license
allowed-tools: Read Write Edit Bash
compatibility: Requires Python 3.11+ and scikit-learn 1.7+. NumPy and SciPy are required dependencies. Optional matplotlib/seaborn for bundled example scripts that save plots.
metadata:
  version: "1.2"
  skill-author: K-Dense Inc.
---

# Scikit-learn & Agentic AutoML Guidelines

## Overview

This skill provides comprehensive guidance for machine learning tasks using scikit-learn, the industry-standard Python library for classical machine learning. Use this skill for classification, regression, clustering, dimensionality reduction, preprocessing, model evaluation, and building production-ready ML pipelines.

**IMPORTANT**: This file also contains project-specific Syntax Guardrails for the Automated Kaggle Pipeline. Always adhere to these guardrails to ensure compatibility with the execution sandbox and `train_model.py`.

---

## Agentic Coding Guidelines & Syntax Guardrails (Project Specific)

You are writing code for an automated machine learning pipeline. To prevent `TypeError`, `NotFittedError`, and API deprecation crashes, you MUST strictly adhere to the following library-specific syntax rules, as outlined in `llms.txt`.

### 1. Model API Strict Syntax (LightGBM, XGBoost, CatBoost)

Different gradient boosting frameworks handle early stopping, verbosity, and multiprocessing differently in their modern versions. You must use the correct API for each.

#### LightGBM (v4.0+)

* **Verbosity:** Set `verbosity=-1` in the constructor. Do not use `verbose`.
* **Early Stopping:** MUST be implemented via `callbacks` in `.fit()`. Do not pass it to the constructor.
* **Multiprocessing:** Use `n_jobs=-1`.

```python
# CORRECT LIGHTGBM SYNTAX
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

model = LGBMClassifier(n_estimators=2000, verbosity=-1, n_jobs=-1, random_state=42)
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
)
```

#### XGBoost (v2.0+)

* **Verbosity:** Pass `verbose=False` into `.fit()`.
* **Early Stopping:** Set `early_stopping_rounds` in the constructor (`XGBClassifier(...)`).
* **Multiprocessing:** Use `n_jobs=-1`.

```python
# CORRECT XGBOOST SYNTAX
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=2000, early_stopping_rounds=50, n_jobs=-1, random_state=42)
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=False
)
```

#### CatBoost (v1.2+)

* **Verbosity:** Set `verbose=0` in the constructor.
* **Early Stopping:** Set `early_stopping_rounds` in the constructor.
* **Multiprocessing:** MUST use `thread_count=-1` (CatBoost does not recognize `n_jobs`).

```python
# CORRECT CATBOOST SYNTAX
from catboost import CatBoostClassifier

model = CatBoostClassifier(n_estimators=2000, early_stopping_rounds=50, verbose=0, thread_count=-1, random_state=42)
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)]
)
```

### 2. Ridge Regression / Classification Constraints

* **Scaling:** Linear models require scaled data. Ensure `StandardScaler` or `RobustScaler` is applied to numerical features before fitting.
* **Probabilities:** `RidgeClassifier` does NOT support `.predict_proba()`. If you need probabilities (e.g., for soft voting or AUC scoring), you must wrap it in a `CalibratedClassifierCV`.

```python
# CORRECT RIDGE CLASSIFIER PROBABILITY SYNTAX
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

base_ridge = RidgeClassifier(random_state=42)
prob_ridge = CalibratedClassifierCV(base_ridge, cv=5)
# prob_ridge now safely supports .predict_proba()
```

### 3. Scikit-Learn (v1.3+) Parameter Quirks

Due to specific scikit-learn version constraints in this environment, the parameter names for disabling sparse matrices differ between preprocessing modules. You must not mix them up:

* **OneHotEncoder**: You MUST use `sparse_output=False`. (Do not use `sparse`).
* **MissingIndicator**: You MUST use `sparse=False`. (Do not use `sparse_output`).

### 4. Pipeline Execution in Cross-Validation

If you build a `Pipeline` or `ColumnTransformer` (especially for feature selection) inside a cross-validation loop, you MUST fit it on the training fold before calling transform.

**INCORRECT**: `X_train_sel = prep_pipeline.transform(X_train)` (Raises NotFittedError)

**CORRECT**:

```python
prep_pipeline.fit(X_train, y_train)
X_train_sel = prep_pipeline.transform(X_train)
X_val_sel = prep_pipeline.transform(X_val)
```

### 5. Pandas Data Type Safety (Avoiding TypeErrors)

When engineering features on string/categorical columns, ALWAYS cast to string before applying string operations or list comprehensions, as columns may contain NaN (which evaluate as floats).

**INCORRECT**: `df[col].apply(lambda x: sum(c.isdigit() for c in x))` (Crashes on float NaN)
**CORRECT**: `df[col].astype(str).apply(lambda x: sum(c.isdigit() for c in x))`
**BETTER**: Use vectorized pandas string methods where possible: `df[col].astype(str).str.contains(r'\d')`

### 6. Kaggle Pipeline Architecture Constraints

As detailed in `llms.txt` and `train_model.py`:

* **Dataset Agnosticism**: The generated `train_model.py` MUST NOT hardcode dataset-specific paths or column names. It must read `dataset_path`, `target_col`, and `test_path` from the configuration file at runtime and support a `--config` argument.
* **Artifact Output**: The generated script MUST accept `--output_dir` (using `argparse`) and write its final cross-validation score to a file named `metrics.json` located within that directory.
* **ID Column Preservation**: When generating submissions, you MUST capture the original ID column (typically the first column of the test set) before dropping it from the feature set.
* **Pipeline Integrity**: The modeling pipeline heavily relies on scikit-learn `Pipeline` and `ColumnTransformer` architectures to prevent data leakage and handle unseen categories safely (`handle_unknown='ignore'`).

### 7. Safe Ensembling & Blending (Preventing Data Leakage)
When building ensembles like `StackingClassifier` or `VotingClassifier`, DO NOT manually call `.fit()` on the base estimators. The scikit-learn meta-estimator handles fitting natively.

**Soft Voting Blending (Highly Recommended):**
A simple soft voting ensemble of diverse GBDT models is often the safest and most effective way to improve CV without overfitting.
```python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

estimators = [
    ('xgb', XGBClassifier(random_state=42, n_jobs=-1, verbose=False)),
    ('lgb', LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)),
    ('cat', CatBoostClassifier(random_state=42, verbose=0, thread_count=-1))
]
# Soft voting averages the predicted probabilities
ensemble = VotingClassifier(estimators=estimators, voting='soft')
# Then place 'ensemble' inside your Pipeline as the classifier
```

---

### 8. The "Single-Axis" Compute Constraint
Execution time is strictly limited (e.g., 600 seconds). You cannot do everything at once. 
When in "Architecture & Tuning Mode", you must choose ONLY ONE of the following strategies per iteration:
1. **Strategy A (Ensembling):** Build a `StackingClassifier` or `VotingClassifier`, but use DEFAULT hyperparameters for the base models. DO NOT perform hyperparameter search inside an ensemble.
2. **Strategy B (Tuning):** Use a SINGLE base model (e.g., LightGBM) and tune it using `RandomizedSearchCV` (strictly set `n_iter=5` or lower). DO NOT use exhaustive `GridSearchCV`.

---

### 9. Known Anti-Patterns (Empirically Proven to Hurt)
- **Do NOT apply PCA before tree-based models.** Trees are invariant to linear transformations — PCA just discards information without helping the trees.
- **Do NOT use `SelectFromModel` with the same estimator class as the final classifier** (e.g., `LGBMClassifier` selector for a `LGBMClassifier`). They learn nearly the same features, adding no value while introducing noise.
- **Do NOT build a `StackingClassifier` with a `LogisticRegression` meta-learner** when the base models already produce well-calibrated probabilities. The meta-learner's linear combination rarely beats a simple average/soft voting.
- **Do NOT wrap feature selection in `FeatureUnion`**. Combining two selectors picks overlapping features, bloating the feature space instead of pruning it.

### 10. Plateau Escape Strategies (When Score Stops Improving)
If you are stuck and the score isn't improving, try these approaches in order:
1. **Tune learning_rate + n_estimators with early stopping** — the single most reliable knob for GBDT models.
2. **Try CatBoost** — often outperforms LightGBM on categorical-heavy datasets with minimal tuning.
3. **Soft VotingClassifier** with LGBM + CatBoost + XGBoost using default params — model diversity beats tuning a single model.
4. **Adjust regularization** (`reg_alpha`, `reg_lambda`, `min_child_samples`) — specifically target overfitting if the training score is much higher than CV.

---

## When to Use This Skill

Use the scikit-learn skill when:

* Building classification or regression models
* Performing clustering or dimensionality reduction
* Preprocessing and transforming data for machine learning
* Evaluating model performance with cross-validation
* Tuning hyperparameters with grid or random search
* Creating ML pipelines for production workflows
* Comparing different algorithms for a task
* Working with both structured (tabular) and text data
* Need interpretable, classical machine learning approaches

## Quick Start

### Complete Pipeline with Mixed Data (Automated Kaggle Compatible)

```python
import argparse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from utils import load_config # Example import from project

# Config loading (NO HARDCODED PATHS)
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--output_dir", type=str, default=".")
args = parser.parse_args()

config = load_config(args.config)
# ... data loading ...

# Define feature types
numeric_features = ['age', 'income']
categorical_features = ['gender', 'occupation']

# Create preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False REQUIRED
])

# Combine transformers
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)) # verbosity=-1 REQUIRED
])

# Fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Core Capabilities

### 1. Supervised Learning

Comprehensive algorithms for classification and regression tasks.

**Key algorithms:**
* **Linear models**: Logistic Regression, Linear Regression, Ridge, Lasso, ElasticNet
* **Tree-based**: Decision Trees, Random Forest, Gradient Boosting
* **Support Vector Machines**: SVC, SVR with various kernels
* **Ensemble methods**: AdaBoost, Voting, Stacking
* **Neural Networks**: MLPClassifier, MLPRegressor
* **Others**: Naive Bayes, K-Nearest Neighbors

### 2. Unsupervised Learning

Discover patterns in unlabeled data through clustering and dimensionality reduction.

**Clustering algorithms:**
* **Partition-based**: K-Means, MiniBatchKMeans
* **Density-based**: DBSCAN, HDBSCAN, OPTICS
* **Hierarchical**: AgglomerativeClustering
* **Probabilistic**: Gaussian Mixture Models
* **Others**: MeanShift, SpectralClustering, BIRCH

**Dimensionality reduction:**
* **Linear**: PCA, TruncatedSVD, NMF
* **Manifold learning**: t-SNE, UMAP, Isomap, LLE, MDS, ClassicalMDS (1.8+)
* **Feature extraction**: FastICA, LatentDirichletAllocation

### 3. Model Evaluation and Selection

Tools for robust model evaluation, cross-validation, and hyperparameter tuning.

**Cross-validation strategies:**
* KFold, StratifiedKFold (classification)
* TimeSeriesSplit (temporal data)
* GroupKFold (grouped samples)

**Hyperparameter tuning:**
* GridSearchCV (exhaustive search)
* RandomizedSearchCV (random sampling)
* HalvingGridSearchCV (successive halving)

**Metrics:**
* **Classification**: accuracy, precision, recall, F1-score, ROC AUC, confusion matrix
* **Regression**: MSE, RMSE, MAE, R², MAPE
* **Clustering**: silhouette score, Calinski-Harabasz, Davies-Bouldin

### 4. Data Preprocessing

Transform raw data into formats suitable for machine learning.

**Scaling and normalization:**
* StandardScaler (zero mean, unit variance)
* MinMaxScaler (bounded range)
* RobustScaler (robust to outliers)
* Normalizer (sample-wise normalization)

**Encoding categorical variables:**
* OneHotEncoder (nominal categories)
* OrdinalEncoder (ordered categories)
* LabelEncoder (target encoding)

**Handling missing values:**
* SimpleImputer (mean, median, most frequent)
* KNNImputer (k-nearest neighbors)
* IterativeImputer (multivariate imputation)

**Feature engineering:**
* PolynomialFeatures (interaction terms)
* KBinsDiscretizer (binning)
* Feature selection (RFE, SelectKBest, SelectFromModel)

### 5. Pipelines and Composition

Build reproducible, production-ready ML workflows.

**Key components:**
* **Pipeline**: Chain transformers and estimators sequentially
* **ColumnTransformer**: Apply different preprocessing to different columns
* **FeatureUnion**: Combine multiple transformers in parallel
* **TransformedTargetRegressor**: Transform target variable

## Best Practices

### Always Use Pipelines

Pipelines prevent data leakage and ensure consistency:

```python
# Good: Preprocessing in pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Bad: Preprocessing outside (can leak information)
X_scaled = StandardScaler().fit_transform(X)
```

### Fit on Training Data Only

Never fit on test data:

```python
# Good
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform

# Bad
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(np.vstack([X_train, X_test]))
```

### Use Stratified Splitting for Classification

Preserve class distribution:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Choose Appropriate Metrics

- Balanced data: Accuracy, F1-score
* Imbalanced data: Precision, Recall, ROC AUC, Balanced Accuracy
* Cost-sensitive: Define custom scorer

## Troubleshooting Common Issues

### ConvergenceWarning

**Issue:** Model didn't converge
**Solution:** Increase `max_iter` or scale features

### Poor Performance on Test Set

**Issue:** Overfitting
**Solution:** Use regularization, cross-validation, or simpler model

### Memory Error with Large Datasets

**Solution:** Use algorithms designed for large data

```python
# Use SGD for large datasets
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()

# Or MiniBatchKMeans for clustering
from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(n_clusters=8, batch_size=100)
```
