import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, mean_squared_error
from lightgbm import LGBMClassifier, LGBMRegressor
from pathlib import Path
from tqdm import tqdm
from utils import load_config, clean_column_names

warnings.filterwarnings('ignore')


def train_and_evaluate(config_path="config.yaml", output_dir="."):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")
    nrows = config.get("nrows", None)

    df = pd.read_csv(dataset_path, nrows=nrows)

    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    X = clean_column_names(X)

    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # 1b. Dataset-agnostic column dropping (IDs and near-empty only)
    try:
        cat_cols_all = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    except TypeError:
        cat_cols_all = X.select_dtypes(include=['object', 'category']).columns.tolist()

    cols_to_drop = []
    for col in X.columns:
        n_unique = X[col].nunique(dropna=False)
        miss_pct = X[col].isnull().mean()
        if n_unique == len(X):                       # Likely ID column
            cols_to_drop.append(col)
        elif miss_pct > 0.95:                         # Near-empty column
            cols_to_drop.append(col)

    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    # Recompute categorical columns after dropping
    try:
        cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    except TypeError:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Generic feature engineering (dataset-agnostic)
    # Extract length, word count, and digit count from all object columns
    for col in cat_cols:
        X[f"{col}_len"] = X[col].astype(str).str.len()
        X[f"{col}_words"] = X[col].astype(str).str.split().str.len()
        X[f"{col}_digits"] = X[col].astype(str).apply(lambda x: sum(c.isdigit() for c in str(x)))

    # Group rare categories into '__OTHER__' based on training frequencies
    rare_maps = {}
    for col in cat_cols:
        if X[col].isnull().all():
            continue
        freq = X[col].value_counts(normalize=True, dropna=True)
        keep = set(freq[freq >= 0.01].index)
        rare_maps[col] = keep
        # Preserve NaN so the imputer can code missingness separately from rare
        mask = X[col].notna() & ~X[col].isin(keep)
        X.loc[mask, col] = "__OTHER__"

    # 3. Identify final feature types after engineering
    try:
        categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    except TypeError:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # 4. Define Preprocessing Pipelines
    numeric_transformer = FeatureUnion(transformer_list=[
        ('imputed', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])),
        ('missing', MissingIndicator(features='all', sparse=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    transformers = []
    if numerical_features:
        transformers.append(('num', numeric_transformer, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    # 5. Model Initialization (LightGBM with default hyperparameters)
    if task == 'classification':
        model = LGBMClassifier(random_state=42)
    else:
        model = LGBMRegressor(random_state=42)

    # 6. Create Full Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 7. Cross Validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Running Cross-Validation (folds=5)...")
    scores = []
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        if task == 'classification':
            y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
        else:
            y_pred = fold_pipeline.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            score = -rmse  # negate so higher is better
        scores.append(score)

    final_score = float(np.mean(scores))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 8. Generate Submission (if test_path is provided)
    if test_path and Path(test_path).exists():
        print("Generating submission...")
        pipeline.fit(X, y)

        test_df = pd.read_csv(test_path, nrows=nrows)
        original_test_columns = test_df.columns.tolist()
        test_df = clean_column_names(test_df)

        # Capture ID column before any manipulation
        test_id_name = original_test_columns[0] if len(original_test_columns) > 0 else "id"
        test_id = test_df.iloc[:, 0] if len(test_df.columns) > 0 else None

        # Apply same column dropping as training
        test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

        # Apply same generic feature engineering
        for col in cat_cols:
            if col in test_df.columns:
                test_df[f"{col}_len"] = test_df[col].astype(str).str.len()
                test_df[f"{col}_words"] = test_df[col].astype(str).str.split().str.len()
                test_df[f"{col}_digits"] = test_df[col].astype(str).apply(lambda x: sum(c.isdigit() for c in str(x)))

        for col in cat_cols:
            if col in test_df.columns and col in rare_maps:
                mask = test_df[col].notna() & ~test_df[col].isin(rare_maps[col])
                test_df.loc[mask, col] = "__OTHER__"

        test_X = test_df.reindex(columns=X.columns)

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        submission = pd.DataFrame({test_id_name: test_id})
        submission[target_col] = preds
        submission.to_csv(output_path / "raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()
    train_and_evaluate(args.config, args.output_dir)