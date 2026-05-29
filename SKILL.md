# Agentic Coding Guidelines & Syntax Guardrails

You are writing code for an automated machine learning pipeline. To prevent `TypeError`, `NotFittedError`, and API deprecation crashes, you MUST strictly adhere to the following library-specific syntax rules.

## 1. Model API Strict Syntax (LightGBM, XGBoost, CatBoost)

Different gradient boosting frameworks handle early stopping, verbosity, and multiprocessing differently in their modern versions. You must use the correct API for each.

### LightGBM (v4.0+)

LightGBM removed early stopping and verbosity arguments from `.fit()`.

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

### XGBoost (v2.0+)

XGBoost moved early stopping to the model constructor.

Verbosity: Pass verbose=False into .fit().

Early Stopping: Set early_stopping_rounds in the constructor (XGBClassifier(...)).

Multiprocessing: Use n_jobs=-1.

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

### CatBoost (v1.2+)

CatBoost accepts these parameters natively in the constructor.

Verbosity: Set verbose=0 in the constructor.

Early Stopping: Set early_stopping_rounds in the constructor.

Multiprocessing: MUST use thread_count=-1 (CatBoost does not recognize n_jobs).

```python
# CORRECT CATBOOST SYNTAX
from catboost import CatBoostClassifier

model = CatBoostClassifier(n_estimators=2000, early_stopping_rounds=50, verbose=0, thread_count=-1, random_state=42)
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)]
)
```

## 2. Ridge Regression / Classification Constraints

If you include Ridge or RidgeClassifier in an ensemble:

Scaling: Linear models require scaled data. Ensure StandardScaler or RobustScaler is applied to numerical features before fitting.

Probabilities: RidgeClassifier does NOT support .predict_proba(). If you need probabilities (e.g., for soft voting or AUC scoring), you must wrap it in a CalibratedClassifierCV.

```python
# CORRECT RIDGE CLASSIFIER PROBABILITY SYNTAX
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

base_ridge = RidgeClassifier(random_state=42)
prob_ridge = CalibratedClassifierCV(base_ridge, cv=5)
# prob_ridge now safely supports .predict_proba()
```

## 3. Scikit-Learn (v1.3+) Parameter Quirks

Due to specific scikit-learn version constraints in this environment, the parameter names for disabling sparse matrices differ between preprocessing modules. You must not mix them up:

OneHotEncoder: You MUST use sparse_output=False. (Do not use sparse).

MissingIndicator: You MUST use sparse=False. (Do not use sparse_output).

## 4. Pipeline Execution in Cross-Validation

If you build a Pipeline or ColumnTransformer (especially for feature selection) inside a cross-validation loop, you MUST fit it on the training fold before calling transform.

INCORRECT: X_train_sel = prep_pipeline.transform(X_train) (Raises NotFittedError)

CORRECT:

```python
prep_pipeline.fit(X_train, y_train)
X_train_sel = prep_pipeline.transform(X_train)
X_val_sel = prep_pipeline.transform(X_val)
```

## 5. Pandas Data Type Safety (Avoiding TypeErrors)

When engineering features on string/categorical columns, ALWAYS cast to string before applying string operations or list comprehensions, as columns may contain NaN (which evaluate as floats).

INCORRECT: df[col].apply(lambda x: sum(c.isdigit() for c in x)) (Crashes on float NaN)

CORRECT: df[col].astype(str).apply(lambda x: sum(c.isdigit() for c in x))

BETTER: Use vectorized pandas string methods where possible: df[col].astype(str).str.contains(r'\d')
