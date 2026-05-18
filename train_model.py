import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_and_evaluate(config_path="config.yaml"):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    # Preserve nrows argument to prevent OOM
    df = pd.read_csv(dataset_path, nrows=None)

    # Basic cleanup
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Sanitize column names (remove special chars, replace spaces)
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]

    # Task detection
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values  # keep as numpy array

    # 2. Drop high-missing and constant columns (based on whole training set)
    drop_cols = []
    for col in X.columns:
        miss_ratio = X[col].isna().mean()
        if miss_ratio > 0.7 or X[col].isna().all() or X[col].nunique(dropna=False) <= 1:
            drop_cols.append(col)
    if drop_cols:
        print(f"Dropping columns: {drop_cols}")
    X = X.drop(columns=drop_cols, errors='ignore')

    # Identify feature types after dropping
    try:
        categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    except TypeError:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # 3. Build preprocessing pipeline with imputation
    # Numeric: median imputation; Categorical: most frequent + one-hot
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # in case anything left
    )

    # 4. Model initialisation with tuned hyperparameters
    models = []
    # XGBoost
    try:
        if task == 'classification':
            xgb = XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                use_label_encoder=False, eval_metric='logloss', verbosity=0
            )
        else:
            xgb = XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )
        models.append(('xgb', xgb))
    except Exception as e:
        print(f"XGBoost not available: {e}")

    # LightGBM
    try:
        if task == 'classification':
            lgb = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbose=-1, force_row_wise=True
            )
        else:
            lgb = LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbose=-1, force_row_wise=True
            )
        models.append(('lgb', lgb))
    except Exception as e:
        print(f"LightGBM not available: {e}")

    # CatBoost
    try:
        if task == 'classification':
            cat = CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6,
                random_seed=42, verbose=0
            )
        else:
            cat = CatBoostRegressor(
                iterations=300, learning_rate=0.05, depth=6,
                random_seed=42, verbose=0
            )
        models.append(('cat', cat))
    except Exception as e:
        print(f"CatBoost not available: {e}")

    # HistGradientBoosting (lightweight sklearn)
    try:
        if task == 'classification':
            hist = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=None,
                random_state=42
            )
        else:
            hist = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=None,
                random_state=42
            )
        models.append(('hist', hist))
    except Exception as e:
        print(f"HistGradientBoosting not available: {e}")

    if not models:
        raise RuntimeError("No models could be initialized.")

    # Soft voting for classification, hard voting for regression? Use soft with weights? 
    # Use soft voting with probability averaging for classification; for regression average predictions.
    if task == 'classification':
        ensemble = VotingClassifier(estimators=models, voting='soft')
    else:
        ensemble = VotingRegressor(estimators=models)

    # 5. Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', ensemble)
    ])

    # 6. Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    print("\nRunning 5-fold CV...")
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone the pipeline for safety (not strictly necessary but clean)
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        if task == 'classification':
            # ROC-AUC
            y_pred_proba = fold_pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
        else:
            # Negative RMSE (higher is better) so that score logic matches "higher=better"
            y_pred = fold_pipeline.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            score = -rmse
        scores.append(score)

    final_score = np.mean(scores)
    print(f"CV Score: {final_score:.6f}")

    # 7. Write metrics
    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 8. Submission generation (if test_path present)
    if test_path and os.path.exists(test_path):
        print("\nGenerating submission...")
        # Fit on full training data
        pipeline.fit(X, y)

        # Read test data (preserve nrows argument in case it's there)
        test_df = pd.read_csv(test_path, nrows=None)

        # Ensure test columns match train (only those used in X)
        common_cols = [c for c in X.columns if c in test_df.columns]
        if len(common_cols) < len(X.columns):
            missing = set(X.columns) - set(common_cols)
            print(f"Warning: Test data missing columns: {missing}")
        test_X = test_df[common_cols].copy()

        # Predict
        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        # Build submission DataFrame
        submission = pd.DataFrame()
        # Use the first column of test_df as identifier (if available)
        if len(test_df.columns) > 0:
            submission[test_df.columns[0]] = test_df.iloc[:, 0]
        submission[target_col] = preds

        submission.to_csv("raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    train_and_evaluate(args.config)