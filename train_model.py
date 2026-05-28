import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, PolynomialFeatures, KBinsDiscretizer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    SelectPercentile,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
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

    # Record base feature types before any engineering
    base_numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    try:
        base_categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    except TypeError:
        base_categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Recompute categorical columns after dropping
    try:
        cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    except TypeError:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Generic feature engineering (dataset-agnostic) -- LOCKED, DO NOT MODIFY
    for col in cat_cols:
        if col not in X.columns:
            continue
        X[f"{col}_len"] = X[col].astype(str).str.len()
        X[f"{col}_words"] = X[col].astype(str).str.split().str.len()
        X[f"{col}_digits"] = X[col].astype(str).apply(lambda x: sum(c.isdigit() for c in str(x)))
        X[f"{col}_digit_ratio"] = X[f"{col}_digits"] / X[f"{col}_len"].replace(0, 1)
        X[f"{col}_has_num"] = X[col].astype(str).str.contains(r'\d', regex=True, na=False).astype(int)
        extracted = X[col].astype(str).str.extract(r'(\d+)', expand=False)
        X[f"{col}_num_extract"] = pd.to_numeric(extracted, errors='coerce')
        X[f"{col}_firstchar"] = X[col].astype(str).str[0]

    # Frequency encoding (fit on train only)
    freq_maps = {}
    for col in cat_cols:
        if col not in X.columns:
            continue
        if X[col].isnull().all():
            continue
        freq = X[col].value_counts(normalize=True, dropna=True)
        freq_maps[col] = freq.to_dict()
        X[f"{col}_freq"] = X[col].map(freq_maps[col])

    # Group rare categories into '__OTHER__' based on training frequencies
    rare_maps = {}
    for col in cat_cols:
        if col not in X.columns:
            continue
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
    derived_numerical_features = [c for c in numerical_features if c not in base_numerical_features]

    # 3b. PRUNE: Drop noisy derived numerical features (near-empty or constant)
    derived_to_drop = []
    for col in derived_numerical_features:
        miss_rate = X[col].isnull().mean()
        n_unique = X[col].nunique(dropna=False)
        if miss_rate > 0.70:
            derived_to_drop.append(col)
        elif n_unique <= 1:
            derived_to_drop.append(col)
        else:
            top_freq = X[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq > 0.99:
                derived_to_drop.append(col)
    if derived_to_drop:
        X = X.drop(columns=derived_to_drop)
        numerical_features = [c for c in numerical_features if c not in derived_to_drop]
        derived_numerical_features = [c for c in derived_numerical_features if c not in derived_to_drop]

    # 3c. Split categorical features by cardinality for mixed encoding
    low_cardinality_cats = [c for c in categorical_features if X[c].nunique() <= 10]
    high_cardinality_cats = [c for c in categorical_features if X[c].nunique() > 10]

    # 4. Define Preprocessing Pipelines (LOCKED)
    base_numeric_transformers = []

    if base_numerical_features:
        base_numeric_transformers.append(
            ('original', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]))
        )
        base_numeric_transformers.append(
            ('bins', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=None))
            ]))
        )
        if len(base_numerical_features) <= 10:
            base_numeric_transformers.append(
                ('poly', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
                ]))
            )

    categorical_transformer_low = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    categorical_transformer_high = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    transformers = []
    if base_numerical_features:
        transformers.append(('base_num', FeatureUnion(transformer_list=base_numeric_transformers), base_numerical_features))
    if derived_numerical_features:
        transformers.append(('derived_num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), derived_numerical_features))
    if low_cardinality_cats:
        transformers.append(('cat_low', categorical_transformer_low, low_cardinality_cats))
    if high_cardinality_cats:
        transformers.append(('cat_high', categorical_transformer_high, high_cardinality_cats))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    # 5. Feature Selection (NEW REPRESENTATION: L1 pre-filter + LightGBM importance selection)
    #    Stage 1: L1‑penalized linear model (keep features with importance above median, gentler initial prune)
    if task == 'classification':
        l1_estimator = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=0.2,                     # slightly less aggressive than 0.1
            max_iter=2000,
            random_state=42,
            n_jobs=-1
        )
    else:
        l1_estimator = Lasso(alpha=0.005, random_state=42, max_iter=2000)

    l1_selector = SelectFromModel(
        estimator=l1_estimator,
        threshold='median',           # keep features above median importance
        max_features=None
    )

    #    Stage 2: LightGBM-based selector – the model itself selects the most predictive features
    if task == 'classification':
        lgb_selector_estimator = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' if len(np.unique(y)) == 2 else None,
            verbosity=-1
        )
    else:
        lgb_selector_estimator = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

    lgb_selector = SelectFromModel(
        estimator=lgb_selector_estimator,
        threshold='median',          # features with importance above median
        max_features=None
    )

    # Pipeline for preprocessing + selection (without final model)
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('variance_thresh', VarianceThreshold(threshold=0.0)),
        ('l1_select', l1_selector),        # moderate initial prune
        ('lgb_select', lgb_selector)       # final alignment to learner
    ])

    # 6. Model configuration with increased capacity and stronger regularization
    num_classes = len(np.unique(y))
    if task == 'classification':
        final_model_base = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary' if num_classes == 2 else 'multiclass',
            n_estimators=10000,
            learning_rate=0.01,             # slightly higher to converge faster
            num_leaves=40,                  # a bit more capacity
            max_depth=6,
            subsample=0.75,
            subsample_freq=5,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            min_child_samples=15,
            class_weight='balanced' if num_classes == 2 else None,
            early_stopping_rounds=100,      # earlier stopping to avoid overfit
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        eval_metric = 'auc' if num_classes == 2 else 'multi_logloss'
    else:
        final_model_base = LGBMRegressor(
            boosting_type='gbdt',
            n_estimators=10000,
            learning_rate=0.01,
            num_leaves=40,
            max_depth=6,
            subsample=0.75,
            subsample_freq=5,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            min_child_samples=15,
            early_stopping_rounds=100,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        eval_metric = 'rmse'

    # 7. Cross Validation with early stopping inside each fold
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Running Cross-Validation (folds=5) with early stopping...")
    scores = []
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit preprocessing + selection on training fold only
        prep_pipeline = clone(preprocessing_pipeline)
        prep_pipeline.fit(X_train, y_train)

        X_train_sel = prep_pipeline.transform(X_train)
        X_val_sel = prep_pipeline.transform(X_val)

        # Further split training data for early stopping (80/20)
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X_train_sel, y_train,
            test_size=0.2,
            stratify=y_train if task == 'classification' else None,
            random_state=42
        )

        model = clone(final_model_base)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_ev, y_ev)],
            eval_metric=eval_metric
        )

        # Predict on validation fold
        if task == 'classification':
            if num_classes == 2:
                y_pred = model.predict_proba(X_val_sel)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            else:
                y_pred = model.predict_proba(X_val_sel)[:, 1]   # keep compliance
                score = roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')
        else:
            y_pred = model.predict(X_val_sel)
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
        # Fit preprocessing on full training data
        prep_pipeline_full = clone(preprocessing_pipeline)
        prep_pipeline_full.fit(X, y)
        X_full_sel = prep_pipeline_full.transform(X)

        # Train final model with early stopping using a split from full data
        X_tr_full, X_ev_full, y_tr_full, y_ev_full = train_test_split(
            X_full_sel, y,
            test_size=0.2,
            stratify=y if task == 'classification' else None,
            random_state=42
        )
        model_final = clone(final_model_base)
        model_final.fit(
            X_tr_full, y_tr_full,
            eval_set=[(X_ev_full, y_ev_full)],
            eval_metric=eval_metric
        )

        # Retrain on full transformed data using the best number of estimators
        best_iter = model_final.best_iteration_
        if best_iter is None or best_iter <= 0:
            print(f"Warning: best_iteration_ = {best_iter}, using model_final directly.")
            model_final_full = model_final
        else:
            model_final_full = clone(final_model_base)
            # Disable early stopping for the final full fit (no eval set)
            model_final_full.set_params(n_estimators=best_iter, early_stopping_rounds=None)
            model_final_full.fit(X_full_sel, y)

        # Process test data
        test_df = pd.read_csv(test_path, nrows=nrows)
        original_test_columns = test_df.columns.tolist()
        test_df = clean_column_names(test_df)

        # Capture ID column before any manipulation
        test_id_name = original_test_columns[0] if len(original_test_columns) > 0 else "id"
        test_id = test_df.iloc[:, 0] if len(test_df.columns) > 0 else None

        # Apply same column dropping as training
        test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

        # Apply same generic feature engineering (LOCKED)
        for col in cat_cols:
            if col in test_df.columns:
                test_df[f"{col}_len"] = test_df[col].astype(str).str.len()
                test_df[f"{col}_words"] = test_df[col].astype(str).str.split().str.len()
                test_df[f"{col}_digits"] = test_df[col].astype(str).apply(lambda x: sum(c.isdigit() for c in str(x)))
                test_df[f"{col}_digit_ratio"] = test_df[f"{col}_digits"] / test_df[f"{col}_len"].replace(0, 1)
                test_df[f"{col}_has_num"] = test_df[col].astype(str).str.contains(r'\d', regex=True, na=False).astype(int)
                extracted = test_df[col].astype(str).str.extract(r'(\d+)', expand=False)
                test_df[f"{col}_num_extract"] = pd.to_numeric(extracted, errors='coerce')
                test_df[f"{col}_firstchar"] = test_df[col].astype(str).str[0]

        for col in cat_cols:
            if col in test_df.columns and col in freq_maps:
                test_df[f"{col}_freq"] = test_df[col].map(freq_maps[col]).fillna(0)

        for col in cat_cols:
            if col in test_df.columns and col in rare_maps:
                mask = test_df[col].notna() & ~test_df[col].isin(rare_maps[col])
                test_df.loc[mask, col] = "__OTHER__"

        # Align columns to training set (automatically drops pruned derived features)
        test_X = test_df.reindex(columns=X.columns)

        test_X_sel = prep_pipeline_full.transform(test_X)

        if task == 'classification':
            if num_classes == 2:
                preds = model_final_full.predict_proba(test_X_sel)[:, 1]
            else:
                preds = model_final_full.predict_proba(test_X_sel)[:, 1]
        else:
            preds = model_final_full.predict(test_X_sel)

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