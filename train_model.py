import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
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

    df = pd.read_csv(dataset_path, nrows=None)
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
    
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw

    # 2. Define Preprocessing Pipeline
    try:
        categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns
    except TypeError:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # 3. Model Initialization (Multi-Model Ensemble)
    models = []
    try:
        if task == 'classification':
            models.append(('xgb', XGBClassifier(random_state=42)))
        else:
            models.append(('xgb', XGBRegressor(random_state=42)))
    except Exception: pass
    try:
        if task == 'classification':
            models.append(('lgb', LGBMClassifier(random_state=42, verbose=-1)))
        else:
            models.append(('lgb', LGBMRegressor(random_state=42, verbose=-1)))
    except Exception: pass
    try:
        if task == 'classification':
            models.append(('cat', CatBoostClassifier(random_state=42, verbose=0)))
        else:
            models.append(('cat', CatBoostRegressor(random_state=42, verbose=0)))
    except Exception: pass
    try:
        if task == 'classification':
            models.append(('hist', HistGradientBoostingClassifier(random_state=42)))
        else:
            models.append(('hist', HistGradientBoostingRegressor(random_state=42)))
    except Exception: pass

    if not models: raise RuntimeError("No models could be initialized.")
    if task == 'classification':
        ensemble = VotingClassifier(estimators=models, voting='soft')
    else:
        ensemble = VotingRegressor(estimators=models)

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', ensemble)])
    
    # 5. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'roc_auc'
    
    print(f"Running Cross-Validation (folds=5)...")
    scores = []
    # Manual loop to show progress
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        
        # Scoring
        if task == 'classification':
            if scoring == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            else:
                from sklearn.metrics import get_scorer
                score = get_scorer(scoring)(fold_pipeline, X_val, y_val)
        else:
            from sklearn.metrics import get_scorer
            score = get_scorer(scoring)(fold_pipeline, X_val, y_val)
            
        scores.append(score)

    final_score = np.mean(scores)
    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 6. Generate Submission (if test_path is provided)
    if test_path and os.path.exists(test_path):
        print("Generating submission...")
        pipeline.fit(X, y)
        test_df = pd.read_csv(test_path)
        
        # Ensure test columns match train columns before preprocessing
        test_X = test_df[X.columns.intersection(test_df.columns)]

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
