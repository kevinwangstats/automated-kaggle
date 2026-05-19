import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.base import clone
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    repo_root = os.environ.get("REPO_ROOT", os.getcwd())
    if config.get("dataset_path") and not os.path.isabs(config.get("dataset_path")):
        config["dataset_path"] = os.path.join(repo_root, config["dataset_path"])
    if config.get("test_path") and not os.path.isabs(config.get("test_path")):
        config["test_path"] = os.path.join(repo_root, config["test_path"])
    return config

def engineer_features(df):
    df = df.copy()
    # Generic imputations for known Titanic-like columns (safe if absent)
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        mode_emb = df['Embarked'].mode()
        if not mode_emb.empty:
            df['Embarked'] = df['Embarked'].fillna(mode_emb[0])
    
    # Extract Title from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].astype(str).str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        df = df.drop(columns=['Name'])
    
    # Family size / IsAlone
    if all(c in df.columns for c in ['SibSp', 'Parch']):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Ticket frequency (group size proxy)
    if 'Ticket' in df.columns:
        ticket_counts = df['Ticket'].value_counts()
        df['TicketFreq'] = df['Ticket'].map(ticket_counts)
        df = df.drop(columns=['Ticket'])
    
    # Cabin deck & indicator
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
        df = df.drop(columns=['Cabin'])
    
    return df

def train_and_evaluate(config_path="config.yaml", output_dir="."):
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")
    nrows = config.get("nrows", None)

    # Load train
    df = pd.read_csv(dataset_path, nrows=nrows)
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Clean column names
    X_raw.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X_raw.columns]

    # Detect and drop ID-like column from train (first col if all unique)
    id_col = None
    if X_raw.iloc[:, 0].nunique() == len(X_raw):
        id_col = X_raw.columns[0]
        X_raw = X_raw.drop(columns=[id_col])

    # Feature engineering
    X = engineer_features(X_raw)

    # Task inference
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # Preprocessor
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # Model definitions
    if task == 'classification':
        counts = np.bincount(y)
        scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1.0

        estimators = [
            ('xgb', XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.2, reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42, n_jobs=1
            )),
            ('lgb', LGBMClassifier(
                n_estimators=400, max_depth=-1, learning_rate=0.03,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                class_weight='balanced',
                random_state=42, verbose=-1, n_jobs=1
            )),
            ('cat', CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.03,
                l2_leaf_reg=3.0, border_count=254, random_strength=1,
                auto_class_weights='Balanced',
                random_seed=42, verbose=0, thread_count=1
            )),
            ('hist', HistGradientBoostingClassifier(
                max_iter=400, max_depth=3, learning_rate=0.03,
                l2_regularization=1.0, class_weight='balanced',
                random_state=42
            ))
        ]
        final_estimator = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs')
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=False
        )
        scoring_fn = roc_auc_score
    else:
        estimators = [
            ('xgb', XGBRegressor(
                n_estimators=400, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=1
            )),
            ('lgb', LGBMRegressor(
                n_estimators=400, max_depth=-1, learning_rate=0.03,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=1
            )),
            ('cat', CatBoostRegressor(
                iterations=400, depth=6, learning_rate=0.03,
                l2_leaf_reg=3.0, border_count=254, random_strength=1,
                random_seed=42, verbose=0, thread_count=1
            )),
            ('hist', HistGradientBoostingRegressor(
                max_iter=400, max_depth=3, learning_rate=0.03,
                l2_regularization=1.0,
                random_state=42
            ))
        ]
        final_estimator = RidgeCV()
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=3,
            n_jobs=-1,
            passthrough=False
        )
        scoring_fn = mean_squared_error

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', ensemble)])

    # Cross-validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        if task == 'classification':
            y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
            score = scoring_fn(y_val, y_pred)
        else:
            y_pred = fold_pipeline.predict(X_val)
            score = -scoring_fn(y_val, y_pred)  # negative MSE for consistency with higher-is-better
        scores.append(score)

    final_score = float(np.mean(scores))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"cv_score": final_score}, f)

    # Submission generation
    if test_path and os.path.exists(test_path):
        print("Generating submission...")
        # Refit on full data
        pipeline.fit(X, y)

        # Load test and preserve original ID column before any mutation
        test_df_raw = pd.read_csv(test_path, nrows=nrows)
        test_id_series = test_df_raw.iloc[:, 0]

        test_df = test_df_raw.copy()
        test_df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in test_df.columns]

        # Drop the same ID column used in train; fallback to first column
        if id_col and id_col in test_df.columns:
            test_X_raw = test_df.drop(columns=[id_col])
        else:
            test_X_raw = test_df.drop(columns=[test_df.columns[0]])

        if target_col in test_X_raw.columns:
            test_X_raw = test_X_raw.drop(columns=[target_col])

        test_X = engineer_features(test_X_raw)
        test_X = test_X.reindex(columns=X.columns, fill_value=0)

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        submission = pd.DataFrame()
        submission[test_id_series.name] = test_id_series.values
        submission[target_col] = preds
        submission.to_csv(os.path.join(output_dir, "raw_submission.csv"), index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()
    train_and_evaluate(args.config, args.output_dir)