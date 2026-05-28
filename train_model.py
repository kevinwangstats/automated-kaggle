import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd()))
    if config.get("dataset_path") and not Path(config.get("dataset_path")).is_absolute():
        config["dataset_path"] = str(repo_root / config["dataset_path"])
    if config.get("test_path") and not Path(config.get("test_path")).is_absolute():
        config["test_path"] = str(repo_root / config["test_path"])
    return config

def _add_features(X_df):
    """Generic dataset-agnostic feature engineering."""
    X = X_df.copy()
    # For categorical/text columns, add length and missing indicator
    for col in X.select_dtypes(include=['object', 'category', 'string']).columns:
        X[f"{col}_len"] = X[col].astype(str).replace('nan', '').str.len()
        X[f"{col}_missing"] = X[col].isnull().astype(int)
    # For numeric columns, add missing indicator
    for col in X.select_dtypes(include=np.number).columns:
        X[f"{col}_missing"] = X[col].isnull().astype(int)
    return X

def train_and_evaluate(config_path="config.yaml", output_dir="."):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")
    nrows = config.get("max_rows")

    df = pd.read_csv(dataset_path, nrows=nrows)
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Sanitize column names
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
    
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # Determine original categorical columns before feature engineering
    cat_cols = list(X.select_dtypes(include=['object', 'category', 'string']).columns)

    # 2. Define Preprocessing Pipeline
    transformers = [
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), make_column_selector(dtype_include=np.number)),
    ]
    if len(cat_cols) > 0:
        transformers.append(
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # 3. Model Initialization (Stacking Ensemble)
    if task == 'classification':
        estimators = [
            ('xgb', XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.1, reg_alpha=0.01, reg_lambda=1.0,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )),
            ('lgb', LGBMClassifier(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.01, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=800, depth=6, learning_rate=0.03,
                l2_leaf_reg=3.0, border_count=254,
                random_seed=42, verbose=False, thread_count=-1,
                loss_function='Logloss'
            )),
            ('ridge', CalibratedClassifierCV(
                RidgeClassifier(random_state=42), method='sigmoid', cv=3
            ))
        ]
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=42),
            passthrough=False,
            stack_method='predict_proba',
            cv=3,
            n_jobs=1
        )
        metric = config.get("metric", "roc_auc")
    else:
        estimators = [
            ('xgb', XGBRegressor(
                n_estimators=800, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.1, reg_alpha=0.01, reg_lambda=1.0,
                random_state=42, n_jobs=-1
            )),
            ('lgb', LGBMRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.01, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1
            )),
            ('cat', CatBoostRegressor(
                iterations=800, depth=6, learning_rate=0.03,
                l2_leaf_reg=3.0, border_count=254,
                random_seed=42, verbose=False, thread_count=-1,
                loss_function='RMSE'
            )),
            ('ridge', Ridge(random_state=42))
        ]
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(random_state=42),
            passthrough=False,
            cv=3,
            n_jobs=1
        )
        metric = config.get("metric", "neg_mean_squared_error")

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[
        ('fe', FunctionTransformer(func=_add_features, validate=False)),
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])
    
    # 5. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    if metric is None:
        metric = 'roc_auc' if task == 'classification' else 'neg_mean_squared_error'
    
    print(f"Running Cross-Validation (folds=5, metric={metric})...")
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=1)
    except Exception as e:
        print(f"Warning: requested metric '{metric}' failed ({e}). Falling back to default.")
        metric = 'roc_auc' if task == 'classification' else 'neg_mean_squared_error'
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric, n_jobs=1)

    final_score = float(np.mean(scores))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 6. Generate Submission (if test_path is provided)
    if test_path and Path(test_path).exists():
        print("Generating submission...")
        pipeline.fit(X, y)
        test_df = pd.read_csv(test_path)
        
        # Preserve original ID column (first column)
        test_id = test_df.iloc[:, 0].copy()
        
        # Prepare test features to match training feature set
        test_X = test_df.copy()
        if target_col in test_X.columns:
            test_X = test_X.drop(columns=[target_col])
        test_X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in test_X.columns]
        test_X = test_X.reindex(columns=X.columns, fill_value=np.nan)

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)
            
        submission = pd.DataFrame({
            test_df.columns[0]: test_id,
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