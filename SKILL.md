# Agentic Coding Guidelines & Syntax Guardrails

You are writing code for an environment with specific library versions. To prevent `TypeError` and `NotFittedError` crashes, you MUST strictly adhere to the following syntax rules:

## 1. LightGBM (v4.0+) Strict Syntax
The `LGBMClassifier.fit()` and `LGBMRegressor.fit()` methods no longer accept `early_stopping_rounds` or `verbose` as direct arguments. 
* **INCORRECT:** `model.fit(X, y, early_stopping_rounds=200, verbose=False)`
* **CORRECT:** Pass these arguments to the model constructor during initialization instead:
  `LGBMClassifier(early_stopping_rounds=200, verbosity=-1)`
* **CORRECT (Callbacks):** Or use callbacks in fit: 
  `from lightgbm import early_stopping, log_evaluation`
  `model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[early_stopping(200), log_evaluation(0)])`

## 2. Scikit-Learn Sparse Parameter Quirks
Due to the specific scikit-learn version in this environment, the parameter names for disabling sparse matrices differ between preprocessing modules:
* **OneHotEncoder:** You MUST use `sparse_output=False`. (Do not use `sparse=False`).
* **MissingIndicator:** You MUST use `sparse=False`. (Do not use `sparse_output=False`).

## 3. Pipeline Execution in Cross-Validation
If you build a `Pipeline` or `ColumnTransformer` inside a cross-validation loop, you MUST fit it on the training fold before calling transform.
* **INCORRECT:** `X_train_sel = prep_pipeline.transform(X_train)` (Raises NotFittedError)
* **CORRECT:** `prep_pipeline.fit(X_train, y_train)`
  `X_train_sel = prep_pipeline.transform(X_train)`

## 4. Pandas Data Type Safety (Avoiding TypeErrors)
When engineering features on string/categorical columns, ALWAYS cast to string before applying string operations or list comprehensions, as columns may contain `NaN` (which are floats).
* **INCORRECT:** `df[col].apply(lambda x: sum(c.isdigit() for c in x))` (Crashes on float NaN)
* **CORRECT:** `df[col].astype(str).apply(lambda x: sum(c.isdigit() for c in x))`
* **CORRECT:** Use vectorized pandas string methods where possible: `df[col].astype(str).str.contains(r'\d')`