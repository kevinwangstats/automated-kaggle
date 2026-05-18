import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def engineer_features(df, max_unique=50):
    """Add string length & word count features, then drop high‑cardinality categoricals."""
    df = df.copy()
    new_cols = []
    # Add string‑based features for every object column
    for col in df.select_dtypes(include=['object', 'category']).columns:
        s = df[col].astype(str)
        df[f"{col}_len"] = s.str.len()
        df[f"{col}_wordcnt"] = s.str.count(' ') + 1
        new_cols.extend([f"{col}_len", f"{col}_wordcnt"])
    # Drop original categorical columns with too many unique values
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique(dropna=False) > max_unique:
            df = df.drop(columns=[col])
    return df

def train_and_evaluate(config_path="config.yaml"):
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    # Read data, preserving nrows argument (None = all rows, but prevents OOM if large)
    df = pd.read_csv(dataset_path, nrows=None)

    # Basic cleanup
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Sanitize column names
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]

    # Task detection
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # Feature engineering (on training features)
    X = engineer_features(X)

    # Drop columns with very high missing rate or constant/ all‑na
    drop_cols = []
    for col in X.columns:
        miss_ratio = X[col].isna().mean()
        if miss_ratio > 0.7 or X[col].isna().all() or X[col].nunique(dropna=False) <= 1:
            drop_cols.append(col)
    if drop_cols:
        print(f"Dropping columns: {drop_cols}")
    X = X.drop(columns=drop_cols, errors='ignore')

    # Identify feature types after all operations
    categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Preprocessing pipelines: imputation with indicators where useful
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Base models with improved hyper‑parameters
    models = []

    # XGBoost
    try:
        if task == 'classification':
            xgb = XGBClassifier(
                n_estimators=500, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                use_label_encoder=False, eval_metric='logloss', verbosity=0
            )
        else:
            xgb = XGBRegressor(
                n_estimators=500, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0
            )
        models.append(('xgb', xgb))
    except Exception as e:
        print(f"XGBoost not available: {e}")

    # LightGBM
    try:
        if task == 'classification':
            lgb = LGBMClassifier(
                n_estimators=500, learning_rate=0.03, num_leaves=15,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                random_state=42, verbose=-1, force_row_wise=True
            )
        else:
            lgb = LGBMRegressor(
                n_estimators=500, learning_rate=0.03, num_leaves=15,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                random_state=42, verbose=-1, force_row_wise=True
            )
        models.append(('lgb', lgb))
    except Exception as e:
        print(f"LightGBM not available: {e}")

    # CatBoost
    try:
        if task == 'classification':
            cat = CatBoostClassifier(
                iterations=500, learning_rate=0.03, depth=6,
                l2_leaf_reg=3, random_seed=42, verbose=0
            )
        else:
            cat = CatBoostRegressor(
                iterations=500, learning_rate=0.03, depth=6,
                l2_leaf_reg=3, random_seed=42, verbose=0
            )
        models.append(('cat', cat))
    except Exception as e:
        print(f"CatBoost not available: {e}")

    # HistGradientBoosting
    try:
        if task == 'classification':
            hist = HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.03, max_depth=6,
                l2_regularization=0.1, random_state=42
            )
        else:
            hist = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.03, max_depth=6,
                l2_regularization=0.1, random_state=42
            )
        models.append(('hist', hist))
    except Exception as e:
        print(f"HistGradientBoosting not available: {e}")

    if not models:
        raise RuntimeError("No models could be initialized.")

    # Build ensemble: Stacking with a linear meta‑learner
    if task == 'classification':
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)
        ensemble = StackingClassifier(
            estimators=models, final_estimator=final_estimator,
            cv=5, stack_method='predict_proba'
        )
    else:
        final_estimator = Ridge(alpha=1.0, random_state=42)
        ensemble = StackingRegressor(
            estimators=models, final_estimator=final_estimator,
            cv=5
        )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', ensemble)
    ])

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    print("\nRunning 5‑fold CV...")
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        if task == 'classification':
            y_pred_proba = fold_pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
        else:
            y_pred = fold_pipeline.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            score = -rmse  # higher = better
        scores.append(score)

    final_score = np.mean(scores)
    print(f"CV Score: {final_score:.6f}")

    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # Submission generation
    if test_path and os.path.exists(test_path):
        print("\nGenerating submission...")
        pipeline.fit(X, y)

        test_df = pd.read_csv(test_path, nrows=None)
        # Apply same feature engineering as training
        test_X = engineer_features(test_df)

        # Drop the same columns as training
        test_X = test_X.drop(columns=[c for c in drop_cols if c in test_X.columns], errors='ignore')

        # Ensure columns align with training (only those used)
        common_cols = [c for c in X.columns if c in test_X.columns]
        if len(common_cols) < len(X.columns):
            missing = set(X.columns) - set(common_cols)
            print(f"Warning: Test data missing columns: {missing}")
        test_X = test_X[common_cols].copy()

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        submission = pd.DataFrame()
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