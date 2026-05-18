import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

def extract_title(name):
    """Extract title from name like 'Braund, Mr. Owen Harris' -> 'Mr.'"""
    try:
        title = re.search(r',\s*([^\.]+)\.', name)
        if title:
            return title.group(1).strip()
        return 'Unknown'
    except:
        return 'Unknown'

def group_rare_titles(df, col, threshold=5):
    """Replace titles with less than threshold occurrences with 'Rare'."""
    counts = df[col].value_counts()
    rare = counts[counts < threshold].index
    df[col] = df[col].replace(rare, 'Rare')
    return df

def engineer_features(X):
    """
    Dataset-agnostic feature engineering.
    Checks existence of specific columns before applying transformations.
    """
    X = X.copy()

    # Title extraction from Name
    if 'Name' in X.columns:
        X['Title'] = X['Name'].apply(extract_title)
        # Group rare titles
        title_counts = X['Title'].value_counts()
        rare_titles = title_counts[title_counts < 5].index
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')
        # Optionally drop original Name column
        X.drop('Name', axis=1, inplace=True)

    # FamilySize
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

    # Fare per person
    if 'Fare' in X.columns and 'FamilySize' in X.columns:
        # Avoid division by zero
        X['FarePerPerson'] = X['Fare'] / X['FamilySize'].clip(lower=1)

    # Age: Note missing values will be handled later by imputation
    # We don't modify Age here, just keep as is

    # Cabin: extract first letter (deck)
    if 'Cabin' in X.columns:
        X['Deck'] = X['Cabin'].astype(str).str[0]  # first character
        # Map unknown (nan) to 'U'
        X['Deck'] = X['Deck'].replace('n', 'U')  # 'n' from NaN
        X['Deck'] = X['Deck'].fillna('U')
        X.drop('Cabin', axis=1, inplace=True)

    # Drop Ticket if present (often not useful)
    if 'Ticket' in X.columns:
        X.drop('Ticket', axis=1, inplace=True)

    # Drop PassengerId if present (not a feature)
    if 'PassengerId' in X.columns:
        X.drop('PassengerId', axis=1, inplace=True)

    # One-hot encoding will be done by the preprocessor, so we just keep categorical columns as strings
    return X

def train_and_evaluate(config_path="config.yaml"):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    df = pd.read_csv(dataset_path, nrows=None)
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    task = 'classification' if y_raw.nunique() < 20 else 'regression'

    # 2. Feature Engineering
    X = engineer_features(X)

    # 3. Preprocessing Pipeline
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # SimpleImputer for numerical: median; for categorical: constant fill 'missing'
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, numerical_features),
            ('cat', cat_pipe, categorical_features)
        ],
        remainder='drop'
    )

    # 4. Model Ensemble with tuned hyperparameters
    models = []
    if task == 'classification':
        # XGBoost
        models.append(('xgb', XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        )))
        # LightGBM
        models.append(('lgb', LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
        )))
        # CatBoost
        models.append(('cat', CatBoostClassifier(
            iterations=500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_state=42, verbose=0
        )))
        # HistGradientBoosting
        models.append(('hist', HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42
        )))
    else:
        # Regression counterparts (if needed)
        models.append(('xgb', XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )))
        models.append(('lgb', LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
        )))
        models.append(('cat', CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, random_state=42, verbose=0
        )))
        models.append(('hist', HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42
        )))

    if not models:
        raise RuntimeError("No models could be initialized.")

    if task == 'classification':
        ensemble = VotingClassifier(estimators=models, voting='soft')
    else:
        ensemble = VotingRegressor(estimators=models)

    # 5. Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])

    # 6. Cross-validation
    # We need StratifiedKFold for imbalanced classification; fallback to KFold for regression
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'roc_auc'
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'neg_root_mean_squared_error'  # expected by most regression tasks

    print(f"Running {cv.get_n_splits()}-fold Cross-Validation...")
    scores = []
    for train_idx, val_idx in tqdm(list(cv.split(X, y_raw)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_raw.iloc[train_idx], y_raw.iloc[val_idx]

        if task == 'classification':
            # Encode target to 0/1 for model fitting
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_val_enc = le.transform(y_val)
        else:
            y_train_enc = y_train
            y_val_enc = y_val
            le = None

        # Clone pipeline and fit
        from sklearn.base import clone
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train, y_train_enc)

        # Predict probabilities or values
        if task == 'classification':
            y_pred_proba = fold_pipe.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val_enc, y_pred_proba)
        else:
            y_pred = fold_pipe.predict(X_val)
            score = -np.sqrt(mean_squared_error(y_val_enc, y_pred))

        scores.append(score)

    final_score = np.mean(scores)
    print(f"Mean CV Score ({scoring}): {final_score:.6f}")

    # Write metrics
    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 7. Generate submission
    if test_path and os.path.exists(test_path):
        print("Generating submission on test set...")
        # Fit on full training data
        if task == 'classification':
            le_full = LabelEncoder()
            y_enc = le_full.fit_transform(y_raw)
        else:
            y_enc = y_raw

        pipeline.fit(X, y_enc)

        # Load test data, apply same feature engineering
        test_df = pd.read_csv(test_path)
        test_X = engineer_features(test_df)

        # Ensure test has same columns as training (after engineering)
        missing_cols = set(X.columns) - set(test_X.columns)
        for col in missing_cols:
            test_X[col] = np.nan  # fill with NaN so imputers handle them
        test_X = test_X[X.columns]  # align column order

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        # Build submission DataFrame: first column from test (usually ID), then predictions
        submission = pd.DataFrame()
        if not test_df.empty:
            first_col_name = test_df.columns[0]
            submission[first_col_name] = test_df[first_col_name]
        submission[target_col] = preds
        submission.to_csv("raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    train_and_evaluate(args.config)