import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
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
    Dataset‑agnostic feature engineering with robust missing value handling.
    All transformations are computed on X (training or test) without target leakage.
    """
    X = X.copy()

    # --- 1. Title from Name ---
    if 'Name' in X.columns:
        X['Title'] = X['Name'].apply(extract_title)
        # Group rare titles (fewer than 5 occurrences)
        title_counts = X['Title'].value_counts()
        rare_titles = title_counts[title_counts < 5].index
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')
        X.drop('Name', axis=1, inplace=True)

    # --- 2. Intelligent Age imputation using groups ---
    if 'Age' in X.columns and X['Age'].isna().any():
        group_cols = []
        for col in ['Pclass', 'Sex', 'Title']:
            if col in X.columns:
                group_cols.append(col)
        if group_cols:
            X['Age'] = X.groupby(group_cols, dropna=False)['Age'].transform(
                lambda x: x.fillna(x.median())
            )
        X['Age'] = X['Age'].fillna(X['Age'].median())

    # --- 3. Embarked imputation (mode per Pclass) ---
    if 'Embarked' in X.columns and X['Embarked'].isna().any():
        if 'Pclass' in X.columns:
            X['Embarked'] = X.groupby('Pclass', dropna=False)['Embarked'].transform(
                lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'S')
            )
        else:
            X['Embarked'] = X['Embarked'].fillna('S')

    # --- 4. Fare imputation (median per Pclass) ---
    if 'Fare' in X.columns and X['Fare'].isna().any():
        if 'Pclass' in X.columns:
            X['Fare'] = X.groupby('Pclass', dropna=False)['Fare'].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            X['Fare'] = X['Fare'].fillna(X['Fare'].median())

    # --- 5. Sex numeric feature (model‑friendly binary) ---
    if 'Sex' in X.columns:
        mode_val = X['Sex'].mode()[0] if not X['Sex'].mode().empty else X['Sex'].iloc[0]
        X['Sex_numeric'] = (X['Sex'] == mode_val).astype(int)

    # --- 6. Family size & loneliness ---
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)

    # --- 7. Fare per person ---
    if 'Fare' in X.columns and 'FamilySize' in X.columns:
        X['FarePerPerson'] = X['Fare'] / X['FamilySize'].clip(lower=1)

    # --- 8. Interaction features ---
    if 'Age' in X.columns and 'Pclass' in X.columns:
        X['Age_Pclass'] = X['Age'] * X['Pclass']
    if 'Fare' in X.columns and 'Pclass' in X.columns:
        X['Fare_Pclass'] = X['Fare'] * X['Pclass']
    if 'Age' in X.columns and 'Sex_numeric' in X.columns:
        X['Age_Sex'] = X['Age'] * X['Sex_numeric']
    if 'Fare' in X.columns and 'Sex_numeric' in X.columns:
        X['Fare_Sex'] = X['Fare'] * X['Sex_numeric']
    if 'Pclass' in X.columns and 'Sex_numeric' in X.columns:
        X['Pclass_Sex'] = X['Pclass'] * X['Sex_numeric']

    # --- 9. Log‑transform Fare (reduces skew) ---
    if 'Fare' in X.columns:
        X['LogFare'] = X['Fare'].apply(lambda x: np.log1p(max(x, 0)))

    # --- 10. Cabin: deck and HasCabin flag ---
    if 'Cabin' in X.columns:
        X['HasCabin'] = X['Cabin'].notna().astype(int)
        X['Deck'] = X['Cabin'].astype(str).str[0]
        X['Deck'] = X['Deck'].replace('n', 'U')   # 'n' from 'nan'
        X['Deck'] = X['Deck'].fillna('U')
        X.drop('Cabin', axis=1, inplace=True)

    # --- 11. Age bands (non‑linear) ---
    if 'Age' in X.columns:
        bins = [-np.inf, 12, 18, 25, 35, 50, 65, np.inf]
        labels = ['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior', 'Elder']
        X['AgeBand'] = pd.cut(X['Age'], bins=bins, labels=labels)

    # --- 12. Fare bands (quantile‑based) ---
    if 'Fare' in X.columns:
        try:
            X['FareBand'] = pd.qcut(X['Fare'], q=4, labels=['Low','Med','High','VHigh'], duplicates='drop')
        except ValueError:
            X['FareBand'] = pd.cut(X['Fare'], bins=4, labels=['Q1','Q2','Q3','Q4'])

    # --- 13. Ticket length (if available) ---
    if 'Ticket' in X.columns:
        X['TicketLength'] = X['Ticket'].astype(str).str.len()

    # --- 14. Drop high‑cardinality / ID columns ---
    for col in ['Ticket', 'PassengerId']:
        if col in X.columns:
            X.drop(col, axis=1, inplace=True)

    return X


def train_and_evaluate(config_path="config.yaml"):
    # 1. Load config & data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    df = pd.read_csv(dataset_path, nrows=None)
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    task = 'classification' if y_raw.nunique() < 20 else 'regression'

    # 2. Feature engineering
    X = engineer_features(X)

    # 3. Preprocessing pipeline (numerical + categorical)
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    num_pipe = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=5, random_state=42)),
        ('scaler', RobustScaler())  # RobustScaler is less sensitive to outliers
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

    # 4. Compute class weights for binary classification
    scale_pos_weight_val = None
    if task == 'classification':
        le_global = LabelEncoder()
        y_num = le_global.fit_transform(y_raw)
        if len(le_global.classes_) == 2:
            count0 = np.sum(y_num == 0)
            count1 = np.sum(y_num == 1)
            if count1 > 0:
                scale_pos_weight_val = count0 / count1

    # 5. Build base models with careful regularisation
    base_models = []
    if task == 'classification':
        # XGBoost
        xgb_params = {
            'n_estimators': 800,
            'learning_rate': 0.02,
            'max_depth': 4,
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
            'n_estimators': 800,
            'learning_rate': 0.02,
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
            'iterations': 800,
            'learning_rate': 0.02,
            'depth': 4,
            'l2_leaf_reg': 5,
            'random_strength': 1.0,
            'random_state': 42,
            'verbose': 0,
            'auto_class_weights': 'Balanced',
        }
        base_models.append(('cat', CatBoostClassifier(**cat_params)))

        # HistGradientBoosting
        base_models.append(('hist', HistGradientBoostingClassifier(
            max_iter=800,
            learning_rate=0.02,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=0.5,
            random_state=42
        )))

        # RandomForest
        base_models.append(('rf', RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )))

        # ExtraTrees
        base_models.append(('et', ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )))

        # Logistic Regression (highly regularised)
        base_models.append(('lr', LogisticRegression(
            max_iter=1000,
            C=0.05,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )))

        # Ensemble: VotingClassifier with soft voting (uniform weights)
        ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=-1
        )

    else:  # regression (kept for completeness, analogous changes)
        base_models.append(('xgb', XGBRegressor(
            n_estimators=800, learning_rate=0.02, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42
        )))
        base_models.append(('lgb', LGBMRegressor(
            n_estimators=800, learning_rate=0.02, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            verbose=-1, n_jobs=-1
        )))
        base_models.append(('cat', CatBoostRegressor(
            iterations=800, learning_rate=0.02, depth=4,
            l2_leaf_reg=3, random_state=42, verbose=0
        )))
        base_models.append(('hist', HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.02, max_depth=4,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=42
        )))
        base_models.append(('rf', RandomForestRegressor(
            n_estimators=500, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )))
        ensemble = VotingRegressor(
            estimators=base_models,
            n_jobs=-1
        )

    # 6. Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])

    # 7. Cross‑validation
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

    # 8. Generate submission (if test_path present)
    if test_path and os.path.exists(test_path):
        print("Generating submission on test set...")
        if task == 'classification':
            le_full = LabelEncoder()
            y_enc = le_full.fit_transform(y_raw)
        else:
            y_enc = y_raw

        pipeline.fit(X, y_enc)

        test_df = pd.read_csv(test_path)
        test_X = engineer_features(test_df)

        # Align test columns with training columns (fill missing with NaN)
        for col in set(X.columns) - set(test_X.columns):
            test_X[col] = np.nan
        test_X = test_X[X.columns]

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

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