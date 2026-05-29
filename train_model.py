import pandas as pd
import numpy as np
import yaml
import json
import os
import argparse
from pathlib import Path
from functools import partial
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import StackingClassifier, StackingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.base import clone
from tqdm import tqdm
import warnings
from utils import load_config, clean_column_names

warnings.filterwarnings('ignore')


def engineer_features(df_input, ref_df=None):
    # (locked – no changes allowed)
    df = df_input.copy()
    meta_src = ref_df if ref_df is not None else df

    # 1. Missing indicators
    for col in df.columns:
        if meta_src[col].isnull().any():
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # 2. Auto-detect boolean-like object columns and convert to 0/1
    bool_map = {
        'true': 1, 'false': 0,
        '1': 1, '0': 0,
        'yes': 1, 'no': 0,
        't': 1, 'f': 0,
        'y': 1, 'n': 0
    }
    bool_cols = []
    for col in meta_src.select_dtypes(include=['object', 'category']).columns:
        non_null = meta_src[col].dropna().astype(str).str.strip().str.lower()
        if len(non_null) == 0:
            continue
        uniq = non_null.unique()
        if len(uniq) <= 2 and all(u in bool_map for u in uniq):
            bool_cols.append(col)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(bool_map).astype(float)

    # Recompute column types after bool conversion
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Generic numerical features
    if len(num_cols) > 0:
        # Clip outliers generically based on train distribution
        for col in num_cols:
            if col in meta_src.columns and pd.api.types.is_numeric_dtype(meta_src[col]):
                q_low = meta_src[col].quantile(0.01)
                q_high = meta_src[col].quantile(0.99)
                if pd.notna(q_low) and pd.notna(q_high):
                    df[col] = df[col].clip(lower=q_low, upper=q_high)

        df['__num_sum'] = df[num_cols].sum(axis=1, skipna=True)
        df['__num_mean'] = df[num_cols].mean(axis=1, skipna=True)
        df['__num_std'] = df[num_cols].std(axis=1, skipna=True)
        df['__num_max'] = df[num_cols].max(axis=1, skipna=True)
        df['__num_min'] = df[num_cols].min(axis=1, skipna=True)
        df['__num_null_count'] = df[num_cols].isnull().sum(axis=1)
        df['__num_zero_count'] = (df[num_cols].fillna(0) == 0).sum(axis=1)
        df['__all_num_zero'] = (df['__num_zero_count'] == len(num_cols)).astype(int)

        # Per-column zero indicator
        for col in num_cols:
            if col in meta_src.columns and pd.api.types.is_numeric_dtype(meta_src[col]):
                zero_pct = (meta_src[col].fillna(0) == 0).mean()
                if zero_pct > 0.05:
                    df[f'{col}_is_zero'] = (df[col].fillna(0) == 0).astype(int)

        # Log1p transform for highly skewed non-negative numeric columns
        for col in num_cols:
            if col in meta_src.columns and pd.api.types.is_numeric_dtype(meta_src[col]):
                if meta_src[col].min() >= 0 and meta_src[col].max() > 0:
                    skewness = meta_src[col].dropna().skew()
                    if skewness > 1.5:
                        df[f'{col}_log1p'] = np.log1p(df[col])

        # Generic age heuristic
        for col in num_cols:
            if col.lower() == 'age' and pd.api.types.is_numeric_dtype(df[col]):
                df[f'{col}_is_child'] = (df[col] < 13).astype(int)
                df[f'{col}_is_teen'] = ((df[col] >= 13) & (df[col] < 20)).astype(int)
                df[f'{col}_is_senior'] = (df[col] >= 60).astype(int)

    # 4. Generic structured categorical splitting + numeric part parsing
    for col in cat_cols:
        df[col] = df[col].astype(str).replace('nan', 'missing')
        df[f'{col}_len'] = df[col].str.len()
        for sep, n_splits in [('/', 2), ('_', 1), (' ', 1)]:
            if df[col].str.contains(sep, regex=False, na=False).mean() > 0.1:
                try:
                    parts = df[col].str.split(sep, n=n_splits, expand=True)
                    n_part = min(parts.shape[1], 3)
                    for i in range(n_part):
                        new_col = f'{col}_part{i}'
                        df[new_col] = parts[i].astype(str)
                        parsed = pd.to_numeric(parts[i], errors='coerce')
                        if parsed.notna().sum() > len(df) * 0.5:
                            df[f'{new_col}_num'] = parsed
                except Exception:
                    pass

    # 5. Frequency encoding for all object columns (including generated parts)
    src = ref_df if ref_df is not None else df
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in obj_cols:
        try:
            if col in src.columns:
                freq_map = src[col].value_counts(dropna=False).to_dict()
            else:
                freq_map = df[col].value_counts(dropna=False).to_dict()
            df[f'{col}_freq'] = df[col].map(freq_map).fillna(0).astype(int)
        except Exception:
            df[f'{col}_freq'] = 0

    # Final safety: coerce any remaining object columns that are purely numeric
    for col in df.select_dtypes(include=['object', 'category']).columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        if coerced.notna().mean() > 0.99:
            df[col] = coerced

    return df


def train_and_evaluate(config_path="config.yaml", output_dir="."):
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    read_csv_kwargs = {}
    if "nrows" in config:
        read_csv_kwargs["nrows"] = config["nrows"]

    df = pd.read_csv(dataset_path, **read_csv_kwargs)
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])
    X_raw = clean_column_names(X_raw)

    # Determine task & encode target
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # Feature Engineering (LOCKED – do not modify)
    X = engineer_features(X_raw)
    X = clean_column_names(X)

    # Separate numeric vs non-numeric
    numerical_features = []
    categorical_features = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numerical_features.append(c)
        else:
            categorical_features.append(c)

    # Preprocessor (LOCKED – do not modify)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # ------------------------------------------------------------
    # IMPROVED FEATURE REPRESENTATION:
    # 1) Mutual Information selects top 50 features (or all if fewer).
    # 2) ExtraTrees-based SelectFromModel prunes to median importance.
    # Reduces noise and retains non‑linear interactions.
    # ------------------------------------------------------------
    if task == 'classification':
        mi_selector = SelectKBest(mutual_info_classif, k=50)
        et_selector = SelectFromModel(
            ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='median'
        )
        scoring_fn = roc_auc_score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        mi_selector = SelectKBest(mutual_info_regression, k=50)
        et_selector = SelectFromModel(
            ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='median'
        )
        scoring_fn = mean_squared_error
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Compose the two feature selection steps
    feature_selector = Pipeline(steps=[
        ('mi', mi_selector),
        ('et', et_selector)
    ])

    # ------------------------------------------------------------

    # Initialize Stacking Ensemble (unchanged hyper‑parameters)
    if task == 'classification':
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier

        estimators = [
            ('lgb', LGBMClassifier(
                n_estimators=2000, learning_rate=0.01, num_leaves=7, max_depth=3,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.6, reg_lambda=0.6,
                min_child_samples=50, random_state=42, n_jobs=-1, verbosity=-1)),
            ('xgb', XGBClassifier(
                n_estimators=2000, learning_rate=0.01, max_depth=3,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.6, reg_lambda=0.6,
                min_child_weight=20, gamma=0.2,
                use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1,
                verbosity=0)),
            ('cat', CatBoostClassifier(
                iterations=2000, learning_rate=0.01, depth=3, l2_leaf_reg=12,
                random_strength=2, bagging_temperature=0.5, border_count=32,
                loss_function='Logloss', random_seed=42, verbose=0, thread_count=-1))
        ]
        final_estimator = LogisticRegression(max_iter=2000, C=0.1, solver='lbfgs', random_state=42)
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1
        )
    else:
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor

        estimators = [
            ('lgb', LGBMRegressor(
                n_estimators=2000, learning_rate=0.01, num_leaves=7, max_depth=3,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.6, reg_lambda=0.6,
                min_child_samples=50, random_state=42, n_jobs=-1, verbosity=-1)),
            ('xgb', XGBRegressor(
                n_estimators=2000, learning_rate=0.01, max_depth=3,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.6, reg_lambda=0.6,
                min_child_weight=20, gamma=0.2,
                random_state=42, n_jobs=-1, verbosity=0)),
            ('cat', CatBoostRegressor(
                iterations=2000, learning_rate=0.01, depth=3, l2_leaf_reg=12,
                random_strength=2, bagging_temperature=0.5, border_count=32,
                random_seed=42, verbose=0, thread_count=-1))
        ]
        final_estimator = Ridge(alpha=1.0, random_state=42)
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            passthrough=False,
            n_jobs=-1
        )

    # Build final pipeline with the new two‑stage feature selection
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

    # Cross-Validation
    print("Running 5-Fold CV...")
    scores = []
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        if task == 'classification':
            y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
            score = scoring_fn(y_val, y_pred)
        else:
            y_pred = fold_pipeline.predict(X_val)
            score = -scoring_fn(y_val, y_pred)
        scores.append(score)

    final_score = float(np.mean(scores))
    print(f"Final CV Score: {final_score:.4f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # Generate Submission
    if test_path and os.path.exists(test_path):
        print("Generating submission...")
        test_df = pd.read_csv(test_path, **read_csv_kwargs)
        test_id = test_df.iloc[:, 0].copy()
        test_X_raw = test_df.copy()
        if target_col in test_X_raw.columns:
            test_X_raw = test_X_raw.drop(columns=[target_col])

        test_X_raw = clean_column_names(test_X_raw)
        test_X = engineer_features(test_X_raw, ref_df=X_raw)
        test_X = test_X.reindex(columns=X.columns, fill_value=0)

        pipeline.fit(X, y)
        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        submission = pd.DataFrame({
            test_df.columns[0]: test_id,
            target_col: preds
        })
        submission.to_csv(out_dir / "raw_submission.csv", index=False)

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()
    train_and_evaluate(args.config, args.output_dir)