import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import clone
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


def engineer_features(X):
    """
    Dataset-agnostic feature engineering.
    Checks existence of specific columns before applying transformations.
    """
    X = X.copy()

    # Title extraction from Name
    if 'Name' in X.columns:
        X['Title'] = X['Name'].apply(extract_title)
        # Group rare titles (fewer than 5 occurrences)
        title_counts = X['Title'].value_counts()
        rare_titles = title_counts[title_counts < 5].index
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')
        X.drop('Name', axis=1, inplace=True)

    # FamilySize
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

    # Fare per person
    if 'Fare' in X.columns and 'FamilySize' in X.columns:
        X['FarePerPerson'] = X['Fare'] / X['FamilySize'].clip(lower=1)

    # Interaction features
    if 'Age' in X.columns and 'Pclass' in X.columns:
        X['Age_Pclass'] = X['Age'] * X['Pclass']
    if 'Fare' in X.columns and 'Pclass' in X.columns:
        X['Fare_Pclass'] = X['Fare'] * X['Pclass']

    # Cabin: extract first letter (deck)
    if 'Cabin' in X.columns:
        X['Deck'] = X['Cabin'].astype(str).str[0]  # first character
        X['Deck'] = X['Deck'].replace('n', 'U')
        X['Deck'] = X['Deck'].fillna('U')
        X.drop('Cabin', axis=1, inplace=True)

    # Age band (non‑linear relationship)
    if 'Age' in X.columns:
        bins = [-np.inf, 12, 18, 25, 35, 50, 65, np.inf]
        labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior', 'Elder']
        X['AgeBand'] = pd.cut(X['Age'], bins=bins, labels=labels)

    # Fare band (non‑linear relationship, quantile based to be robust)
    if 'Fare' in X.columns:
        try:
            X['FareBand'] = pd.qcut(X['Fare'], q=4, labels=['Low','Med','High','VHigh'])
        except ValueError:
            # fallback if not enough distinct values
            X['FareBand'] = pd.cut(X['Fare'], bins=4, labels=['Q1','Q2','Q3','Q4'])

    # Drop Ticket if present
    if 'Ticket' in X.columns:
        X.drop('Ticket', axis=1, inplace=True)

    # Drop PassengerId if present
    if 'PassengerId' in X.columns:
        X.drop('PassengerId', axis=1, inplace=True)

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

    # Use IterativeImputer for numerical features to better handle missing values
    num_pipe = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
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

    # 4. Class imbalance handling (for binary classification)
    scale_pos_weight_val = None
    if task == 'classification':
        le_global = LabelEncoder()
        y_num = le_global.fit_transform(y_raw)
        n_classes = len(le_global.classes_)
        if n_classes == 2:  # binary
            count0 = np.sum(y_num == 0)
            count1 = np.sum(y_num == 1)
            if count1 > 0:
                scale_pos_weight_val = count0 / count1

    # 5. Base models with tuned hyperparameters
    base_models = []
    if task == 'classification':
        # XGBoost
        xgb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }
        if scale_pos_weight_val is not None:
            xgb_params['scale_pos_weight'] = scale_pos_weight_val
        base_models.append(('xgb', XGBClassifier(**xgb_params)))

        # LightGBM
        lgb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        if scale_pos_weight_val is not None:
            lgb_params['scale_pos_weight'] = scale_pos_weight_val
        base_models.append(('lgb', LGBMClassifier(**lgb_params)))

        # CatBoost
        cat_params = {
            'iterations': 1000,
            'learning_rate': 0.01,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_state': 42,
            'verbose': 0,
            'auto_class_weights': 'Balanced',
        }
        base_models.append(('cat', CatBoostClassifier(**cat_params)))

        # HistGradientBoosting
        base_models.append(('hist', HistGradientBoostingClassifier(
            max_iter=1000,
            learning_rate=0.01,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42
        )))

        # RandomForest
        base_models.append(('rf', RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )))

        # LogisticRegression (adds linear capability)
        base_models.append(('lr', LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )))

        # Stacking with a regularised LogisticRegression meta‑learner
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight='balanced',
                random_state=42
            ),
            stack_method='predict_proba',
            cv=5,
            n_jobs=-1
        )
    else:  # regression
        base_models.append(('xgb', XGBRegressor(
            n_estimators=1000, learning_rate=0.01, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42
        )))
        base_models.append(('lgb', LGBMRegressor(
            n_estimators=1000, learning_rate=0.01, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            verbose=-1, n_jobs=-1
        )))
        base_models.append(('cat', CatBoostRegressor(
            iterations=1000, learning_rate=0.01, depth=6,
            l2_leaf_reg=3, random_state=42, verbose=0
        )))
        base_models.append(('hist', HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.01, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=42
        )))
        base_models.append(('rf', RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )))
        ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0, random_state=42),
            cv=5,
            n_jobs=-1
        )

    # 6. Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])

    # 7. Cross-validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'roc_auc'
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'neg_root_mean_squared_error'

    print(f"Running {cv.get_n_splits()}-fold Cross-Validation...")
    scores = []
    for train_idx, val_idx in tqdm(list(cv.split(X, y_raw)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_raw.iloc[train_idx], y_raw.iloc[val_idx]

        if task == 'classification':
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_val_enc = le.transform(y_val)
        else:
            y_train_enc = y_train
            y_val_enc = y_val
            le = None

        # Clone and fit
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train, y_train_enc)

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

    # 8. Generate submission
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

        # Align columns with training data (fill missing columns with NaN)
        for col in set(X.columns) - set(test_X.columns):
            test_X[col] = np.nan
        test_X = test_X[X.columns]

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