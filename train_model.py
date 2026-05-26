import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.metrics import roc_auc_score, mean_squared_error
from pathlib import Path
from tqdm import tqdm
from utils import load_config, clean_column_names

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


class GenericFeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self, cat_rare_threshold=20):
        self.cat_rare_threshold = cat_rare_threshold
        self.missing_cols = []
        self.cat_cols = []
        self.num_cols = []
        self.freq_maps = {}
        self.rare_maps = {}
        self.num_medians = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.missing_cols = [c for c in X.columns if X[c].isnull().any()]
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()

        for col in self.cat_cols:
            vc = X[col].value_counts(dropna=False)
            self.freq_maps[col] = vc.to_dict()
            if len(vc) > self.cat_rare_threshold:
                keep = set(vc.head(self.cat_rare_threshold).index.tolist())
            else:
                keep = set(vc.index.tolist())
            self.rare_maps[col] = keep

        for col in self.num_cols:
            self.num_medians[col] = X[col].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Missingness indicators
        for col in self.missing_cols:
            if col in X.columns:
                X[f"{col}_is_missing"] = X[col].isnull().astype(np.int8)
            else:
                X[f"{col}_is_missing"] = 0

        # Categorical engineering
        for col in self.cat_cols:
            if col not in X.columns:
                continue
            keep_vals = self.rare_maps.get(col, set())
            is_common_or_null = X[col].isin(keep_vals) | X[col].isnull()
            X[col] = X[col].where(is_common_or_null, '__rare__')

            # String-derived numeric features
            if X[col].notna().any():
                str_ser = X[col].astype(object).fillna('').astype(str).replace(['nan', 'None', '<NA>', 'NaN'], '')
                X[f"{col}_len"] = str_ser.str.len()
                X[f"{col}_wc"] = str_ser.str.split().str.len().fillna(0)
            else:
                X[f"{col}_len"] = 0
                X[f"{col}_wc"] = 0

            # Frequency encoding
            X[f"{col}_freq"] = X[col].map(self.freq_maps.get(col, {})).fillna(0)

            # FIX: Convert NaN to explicit string placeholder so CatBoost accepts the column
            X[col] = X[col].fillna('__missing__')

        # Numeric imputation
        for col in self.num_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.num_medians.get(col, 0))

        # Generic numeric interactions (original numerics only)
        if len(self.num_cols) >= 2:
            for i in range(len(self.num_cols)):
                for j in range(i + 1, len(self.num_cols)):
                    c1, c2 = self.num_cols[i], self.num_cols[j]
                    if c1 in X.columns and c2 in X.columns:
                        X[f"{c1}_mul_{c2}"] = X[c1] * X[c2]
                        denom = X[c2].replace(0, np.nan)
                        X[f"{c1}_div_{c2}"] = (X[c1] / denom).fillna(0)

        return X


def train_and_evaluate(config_path="config.yaml", output_dir="."):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    nrows = config.get("nrows", None)
    df = pd.read_csv(dataset_path, nrows=nrows)

    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    X = clean_column_names(X)

    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
        n_classes = len(np.unique(y))
    else:
        y = y_raw.values
        n_classes = None

    # 2. Initialize models
    models = {}
    if task == 'classification':
        if CATBOOST_AVAILABLE:
            models['catboost'] = CatBoostClassifier(
                iterations=2000,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=3,
                random_strength=1,
                bagging_temperature=0.5,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=150,
                loss_function='Logloss'
            )
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=2000,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
    else:
        if CATBOOST_AVAILABLE:
            models['catboost'] = CatBoostRegressor(
                iterations=2000,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=150
            )
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBRegressor(
                n_estimators=2000,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )

    if not models:
        raise ImportError("No suitable models are available.")

    # 3. Cross Validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    print(f"Running Cross-Validation (folds=5)...")

    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train_raw, X_val_raw = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        prep = GenericFeatureEngineer(cat_rare_threshold=20)
        X_train = prep.fit_transform(X_train_raw)
        X_val = prep.transform(X_val_raw)

        cat_features = [c for c in prep.cat_cols if c in X_train.columns]

        fold_preds = []

        # CatBoost
        if 'catboost' in models:
            model_cb = clone(models['catboost'])
            model_cb.fit(
                X_train, y_train,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                verbose=False
            )
            if task == 'classification':
                if n_classes == 2:
                    fold_preds.append(model_cb.predict_proba(X_val)[:, 1])
                else:
                    fold_preds.append(model_cb.predict_proba(X_val)[:, 1])
            else:
                fold_preds.append(model_cb.predict(X_val))

        # XGBoost (drop raw categoricals; use engineered numerics only)
        if 'xgboost' in models:
            drop_cols = [c for c in cat_features if c in X_train.columns]
            X_train_xgb = X_train.drop(columns=drop_cols, errors='ignore')
            X_val_xgb = X_val.drop(columns=drop_cols, errors='ignore')

            model_xgb = clone(models['xgboost'])
            model_xgb.fit(
                X_train_xgb, y_train,
                eval_set=[(X_val_xgb, y_val)],
                verbose=False
            )
            if task == 'classification':
                if n_classes == 2:
                    fold_preds.append(model_xgb.predict_proba(X_val_xgb)[:, 1])
                else:
                    fold_preds.append(model_xgb.predict_proba(X_val_xgb)[:, 1])
            else:
                fold_preds.append(model_xgb.predict(X_val_xgb))

        # Ensemble averaging
        if len(fold_preds) > 1:
            y_pred = np.mean(fold_preds, axis=0)
        else:
            y_pred = fold_preds[0]

        if task == 'classification':
            score = roc_auc_score(y_val, y_pred)
        else:
            score = mean_squared_error(y_val, y_pred, squared=False)

        scores.append(score)

    final_score = float(np.mean(scores))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 4. Generate Submission
    if test_path and Path(test_path).exists():
        print("Generating submission...")

        test_df_raw = pd.read_csv(test_path)
        id_col_name = test_df_raw.columns[0]
        id_values = test_df_raw.iloc[:, 0].copy()

        test_df = test_df_raw.copy()
        test_df = clean_column_names(test_df)
        test_X = test_df[X.columns.intersection(test_df.columns)]

        prep_full = GenericFeatureEngineer(cat_rare_threshold=20)
        X_full = prep_full.fit_transform(X)
        test_X_t = prep_full.transform(test_X)
        cat_features = [c for c in prep_full.cat_cols if c in X_full.columns]

        submission_preds = []

        if 'catboost' in models:
            model_cb_full = clone(models['catboost'])
            model_cb_full.fit(X_full, y, cat_features=cat_features, verbose=False)
            if task == 'classification':
                submission_preds.append(model_cb_full.predict_proba(test_X_t)[:, 1])
            else:
                submission_preds.append(model_cb_full.predict(test_X_t))

        if 'xgboost' in models:
            drop_cols = [c for c in cat_features if c in X_full.columns]
            X_full_xgb = X_full.drop(columns=drop_cols, errors='ignore')
            test_X_xgb = test_X_t.drop(columns=drop_cols, errors='ignore')
            model_xgb_full = clone(models['xgboost'])
            model_xgb_full.fit(X_full_xgb, y, verbose=False)
            if task == 'classification':
                submission_preds.append(model_xgb_full.predict_proba(test_X_xgb)[:, 1])
            else:
                submission_preds.append(model_xgb_full.predict(test_X_xgb))

        if len(submission_preds) > 1:
            preds = np.mean(submission_preds, axis=0)
        else:
            preds = submission_preds[0]

        submission = pd.DataFrame({
            id_col_name: id_values,
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