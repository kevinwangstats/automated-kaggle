import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, mean_squared_error
from lightgbm import LGBMClassifier, LGBMRegressor
from pathlib import Path
from tqdm import tqdm
from utils import load_config, clean_column_names

warnings.filterwarnings('ignore')


def add_engineered_features(df):
    """Apply generic, dataset-agnostic feature engineering."""
    df = df.copy()

    # Global missing counts
    df['__missing_count'] = df.isnull().sum(axis=1)

    # Separate column types (exclude already engineered columns)
    obj_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if not c.startswith('__')]

    # Missing counts by dtype
    if len(num_cols) > 0:
        df['__missing_count_num'] = df[num_cols].isnull().sum(axis=1)
    else:
        df['__missing_count_num'] = 0

    if len(obj_cols) > 0:
        df['__missing_count_cat'] = df[obj_cols].isnull().sum(axis=1)
    else:
        df['__missing_count_cat'] = 0

    # String features for every object column
    for col in obj_cols:
        if col.startswith('__'):
            continue
        s = df[col].astype(str)
        df[f'__len_{col}'] = s.str.len()
        df[f'__words_{col}'] = s.str.split().str.len()
        df[f'__digits_{col}'] = s.str.count(r'\d')
        df[f'__upper_{col}'] = s.str.count(r'[A-Z]')
        # Frequency encoding (including NaNs)
        vc = df[col].value_counts(dropna=False)
        df[f'__freq_{col}'] = df[col].map(vc)

    # Generic numerical row-level aggregations
    if len(num_cols) > 0:
        df['__zero_count'] = (df[num_cols] == 0).sum(axis=1)
        df['__negative_count'] = (df[num_cols] < 0).sum(axis=1)
        df['__num_mean'] = df[num_cols].mean(axis=1)
        df['__num_std'] = df[num_cols].std(axis=1, ddof=0)
        df['__num_min'] = df[num_cols].min(axis=1)
        df['__num_max'] = df[num_cols].max(axis=1)
        df['__num_range'] = df['__num_max'] - df['__num_min']

    return df


def train_and_evaluate(config_path="config.yaml", output_dir="."):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")
    nrows = config.get("nrows", None)
    test_nrows = config.get("test_nrows", None)

    df = pd.read_csv(dataset_path, nrows=nrows)
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    X = clean_column_names(X)

    # Convert booleans to integers so they are treated as numerical
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # Drop likely ID columns (high-cardinality integers) to prevent overfitting
    id_like_cols = []
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col]):
            if X[col].nunique() / len(X) > 0.95:
                id_like_cols.append(col)
    if id_like_cols:
        X = X.drop(columns=id_like_cols)

    X = add_engineered_features(X)
    
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # 2. Define Preprocessing Pipeline
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    transformers = []

    # Numerical branch 1: passthrough raw values (LightGBM handles NaNs natively)
    if len(numerical_features) > 0:
        transformers.append(('num_raw', 'passthrough', numerical_features))

    # Numerical branch 2: KNN imputation + quantile binning (memory-gated)
    if len(numerical_features) > 0:
        if X.shape[0] * len(numerical_features) <= 200_000:
            num_imp = KNNImputer(n_neighbors=5)
        else:
            num_imp = SimpleImputer(strategy='median')
        transformers.append(('num_bins', Pipeline(steps=[
            ('imputer', num_imp),
            ('bins', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile'))
        ]), numerical_features))

    # Numerical branch 3: polynomial interactions (only on small memory footprint)
    if len(numerical_features) <= 15:
        transformers.append(('num_poly', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
        ]), numerical_features))

    # Categorical branch: explicit missing level + OHE with cardinality guards
    if len(categorical_features) > 0:
        transformers.append(('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50, min_frequency=0.005))
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # 3. Model Initialization (LightGBM defaults only)
    if task == 'classification':
        model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    else:
        model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # 5. Cross Validation
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
        
        # Scoring
        if task == 'classification':
            y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
        else:
            y_pred = fold_pipeline.predict(X_val)
            score = -np.sqrt(mean_squared_error(y_val, y_pred))
            
        scores.append(score)

    final_score = np.mean(scores)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 6. Generate Submission (if test_path is provided)
    if test_path and Path(test_path).exists():
        print("Generating submission...")
        pipeline.fit(X, y)
        test_df = pd.read_csv(test_path, nrows=test_nrows)
        
        # Capture ID from the first column before any feature manipulation
        id_col_name = test_df.columns[0]
        id_series = test_df.iloc[:, 0]
        
        # Mirror training preprocessing on test set
        test_X_raw = test_df.drop(columns=[target_col], errors='ignore')
        test_X_raw = clean_column_names(test_X_raw)

        # Convert booleans to int
        bool_cols_test = test_X_raw.select_dtypes(include=['bool']).columns
        if len(bool_cols_test) > 0:
            test_X_raw[bool_cols_test] = test_X_raw[bool_cols_test].astype(int)

        # Drop same ID-like columns seen in training
        test_X_raw = test_X_raw.drop(columns=[c for c in id_like_cols if c in test_X_raw.columns], errors='ignore')

        test_X = add_engineered_features(test_X_raw)
        test_X = test_X.reindex(columns=X.columns, fill_value=0)

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)
            
        submission = pd.DataFrame({
            id_col_name: id_series,
            target_col: preds
        })
        submission.to_csv(output_path / "raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()
    train_and_evaluate(args.config, args.output_dir)