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
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, mutual_info_regression, VarianceThreshold
from lightgbm import LGBMClassifier, LGBMRegressor
from pathlib import Path
from tqdm import tqdm
from utils import load_config, clean_column_names

warnings.filterwarnings('ignore')


def add_engineered_features(df):
    """Apply generic, dataset-agnostic feature engineering."""
    df = df.copy()
    original_cols = [c for c in df.columns if not c.startswith('__')]

    # Global missing counts
    df['__missing_count'] = df.isnull().sum(axis=1)

    # Separate column types (exclude already engineered columns)
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    obj_cols = [c for c in obj_cols if not c.startswith('__')]
    num_cols_initial = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_initial = [c for c in num_cols_initial if not c.startswith('__')]

    # Missing counts by dtype
    if len(num_cols_initial) > 0:
        df['__missing_count_num'] = df[num_cols_initial].isnull().sum(axis=1)
    else:
        df['__missing_count_num'] = 0

    if len(obj_cols) > 0:
        df['__missing_count_cat'] = df[obj_cols].isnull().sum(axis=1)
    else:
        df['__missing_count_cat'] = 0

    # Per-column missing indicators (only when informative)
    for col in original_cols:
        missing = df[col].isnull()
        if missing.any() and not missing.all():
            df[f'__missing_{col}'] = missing.astype(int)

    # String features for every object column
    for col in obj_cols:
        if col.startswith('__'):
            continue
        s = df[col].astype(str)
        df[f'__len_{col}'] = s.str.len()
        df[f'__words_{col}'] = s.str.split().str.len()
        df[f'__digits_{col}'] = s.str.count(r'\d')
        df[f'__upper_{col}'] = s.str.count(r'[A-Z]')
        df[f'__has_digit_{col}'] = s.str.contains(r'\d', na=False).astype(int)
        df[f'__title_{col}'] = s.str.count(r'\b\w+\.')
        paren_extract = s.str.extract(r'\((.*?)\)', expand=False)
        df[f'__paren_len_{col}'] = paren_extract.str.len().fillna(0)
        first_word = s.str.split(r'[,\s]+', n=1, expand=False).str[0]
        df[f'__first_word_len_{col}'] = first_word.str.len()
        df[f'__special_{col}'] = s.str.count(r'[^a-zA-Z0-9\s]')
        df[f'__starts_digit_{col}'] = s.str.match(r'^\d').astype(int)
        # Frequency encoding (including NaNs)
        vc = df[col].value_counts(dropna=False)
        df[f'__freq_{col}'] = df[col].map(vc).fillna(0)

        # Extract first title-like token (word ending with period)
        first_title = s.str.extract(r'\b([A-Z][A-Za-z]*)\.', expand=False)
        df[f'__first_title_{col}'] = first_title

        # Extract last word
        last_word = s.str.extract(r'(\b\w+)\W*$', expand=False)
        df[f'__last_word_{col}'] = last_word

        # Ratio of uppercase letters
        caps_ratio = s.str.count(r'[A-Z]') / s.str.len()
        df[f'__caps_ratio_{col}'] = caps_ratio.fillna(0)

        # Punctuation / uniqueness counts (dataset-agnostic proxies for structure)
        df[f'__num_commas_{col}'] = s.str.count(r',')
        df[f'__num_periods_{col}'] = s.str.count(r'\.')
        df[f'__num_slashes_{col}'] = s.str.count(r'/')
        df[f'__num_dashes_{col}'] = s.str.count(r'-')
        df[f'__nunique_chars_{col}'] = s.apply(lambda x: len(set(x)) if isinstance(x, str) else 0)

        # NEW: First / last character (e.g., deck letter, ticket class)
        df[f'__first_char_{col}'] = s.str[0]
        df[f'__last_char_{col}'] = s.str[-1]

        # NEW: Leading non-numeric prefix
        prefix = s.str.extract(r'^([^\d\s]+)', expand=False)
        df[f'__prefix_{col}'] = prefix

        # NEW: Extract first and last numbers as float (ticket number, room number, etc.)
        num_tokens = s.str.replace(r'[^\d]', ' ', regex=True).str.split()
        df[f'__first_num_{col}'] = pd.to_numeric(num_tokens.str[0], errors='coerce')
        df[f'__last_num_{col}'] = pd.to_numeric(num_tokens.str[-1], errors='coerce')

        # NEW: Count numeric groups and tokens
        df[f'__n_numeric_groups_{col}'] = s.str.count(r'\d+')
        df[f'__n_tokens_{col}'] = s.str.split(r'[^a-zA-Z0-9]+').str.len()

        # NEW: Parenthesis presence flag
        df[f'__has_paren_{col}'] = s.str.contains(r'\(', na=False).astype(int)

    # Generic numerical row-level aggregations
    if len(num_cols_initial) > 0:
        df['__zero_count'] = (df[num_cols_initial] == 0).sum(axis=1)
        df['__negative_count'] = (df[num_cols_initial] < 0).sum(axis=1)
        df['__num_mean'] = df[num_cols_initial].mean(axis=1)
        df['__num_std'] = df[num_cols_initial].std(axis=1, ddof=0)
        df['__num_min'] = df[num_cols_initial].min(axis=1)
        df['__num_max'] = df[num_cols_initial].max(axis=1)
        df['__num_range'] = df['__num_max'] - df['__num_min']
        df['__num_median'] = df[num_cols_initial].median(axis=1)
        df['__num_pos_count'] = (df[num_cols_initial] > 0).sum(axis=1)
        df['__num_zero_ratio'] = df['__zero_count'] / len(num_cols_initial)

        # Monotonic transforms for non-negative numerical columns
        for col in num_cols_initial:
            if df[col].min() >= 0:
                df[f'__log1p_{col}'] = np.log1p(df[col])
                df[f'__sqrt_{col}'] = np.sqrt(df[col])
            # Flag integer-valued numbers (helps tree detect pseudo-categoricals)
            # Cast to float so NaN stays NaN and the column is purely numeric,
            # preventing mixed-type (bool/str) failures in sklearn encoders.
            df[f'__is_int_{col}'] = (df[col].mod(1) == 0).where(df[col].notna(), np.nan).astype(float)

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

    # Store core column types before generic engineering (for pairwise/groupby on original features)
    core_numerical = X.select_dtypes(include=[np.number]).columns.tolist()
    core_categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply generic feature engineering
    X = add_engineered_features(X)

    # NEW: Add rank features for all numerical columns present after generic engineering
    # This applies to original numericals and all numerical features generated by add_engineered_features.
    all_numerical_cols_after_generic_fe = X.select_dtypes(include=np.number).columns.tolist()
    for col in all_numerical_cols_after_generic_fe:
        if X[col].nunique() > 1: # Only add rank if there's actual variation for a meaningful rank
            X[f'__rank_{col}'] = X[col].rank(method='average', na_option='keep')
    
    # Pairwise interactions on core numerical columns (dataset-agnostic)
    if 1 < len(core_numerical) <= 15: # Limit to avoid feature explosion
        for i in range(len(core_numerical)):
            for j in range(i + 1, len(core_numerical)):
                c1, c2 = core_numerical[i], core_numerical[j]
                X[f'__interact_{c1}_x_{c2}'] = X[c1] * X[c2]
                X[f'__sum_{c1}_{c2}'] = X[c1] + X[c2]
                X[f'__diff_{c1}_{c2}'] = X[c1] - X[c2]
                # Safe ratio: replace 0 with nan so LightGBM can handle it natively
                X[f'__ratio_{c1}_div_{c2}'] = X[c1] / X[c2].replace(0, np.nan)
                X[f'__ratio_{c2}_div_{c1}'] = X[c2] / X[c1].replace(0, np.nan)

    # Dataset-agnostic groupby statistics & cross-categorical interactions
    groupby_maps = {}
    cat_interact_maps = {}

    # Groupby numerical stats by categorical columns (training set only)
    # Check for reasonable number of categories and numerical features to avoid explosion
    if len(core_categorical) * len(core_numerical) <= 500: 
        for cat_col in core_categorical:
            nuniq = X[cat_col].nunique()
            if 2 <= nuniq <= 100: # Ensure categorical column is not too high/low cardinality
                for num_col in core_numerical:
                    grp = X.groupby(cat_col)[num_col].agg(['mean', 'std', 'median'])
                    groupby_maps[(cat_col, num_col)] = grp
                    X[f'__grp_mean_{cat_col}_{num_col}'] = X[cat_col].map(grp['mean'])
                    X[f'__grp_std_{cat_col}_{num_col}'] = X[cat_col].map(grp['std'])
                    X[f'__grp_median_{cat_col}_{num_col}'] = X[cat_col].map(grp['median'])

    # Cross-categorical interaction frequencies
    if 1 < len(core_categorical) <= 10: # Limit to avoid feature explosion
        for i in range(len(core_categorical)):
            for j in range(i + 1, len(core_categorical)):
                c1, c2 = core_categorical[i], core_categorical[j]
                interact = X[c1].astype(str) + '__' + X[c2].astype(str)
                vc = interact.value_counts(dropna=False)
                cat_interact_maps[(c1, c2)] = vc
                X[f'__freq_interact_{c1}_{c2}'] = interact.map(vc).fillna(0)
    
    # SANITIZE: ensure object columns contain only strings or NaN (no mixed bool/str)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)

    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # 2. Define Preprocessing Pipeline
    # Re-evaluate categorical and numerical features after all engineering
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Cardinality-aware categorical split
    cat_low = [c for c in categorical_features if X[c].nunique() <= 10]
    cat_high = [c for c in categorical_features if X[c].nunique() > 10]

    transformers = []

    # Numerical branch 1: passthrough raw values (LightGBM handles NaNs natively)
    if len(numerical_features) > 0:
        transformers.append(('num_raw', 'passthrough', numerical_features))

    # Numerical branch 2: quantile binning on core numericals only (memory-safe)
    if len(core_numerical) > 0:
        transformers.append(('num_bins_q', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('bins', KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile', subsample=None))
        ]), core_numerical))

    # Numerical branch 3: uniform binning on core numericals
    if len(core_numerical) > 0:
        transformers.append(('num_bins_u', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('bins', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', subsample=None))
        ]), core_numerical))

    # Numerical branch 4: KNN imputation for small datasets (alternative to median for dense features)
    # Apply to *all* numerical features (original + engineered), not just core_numerical
    if len(numerical_features) > 0 and len(X) <= 20000: # Small dataset constraint
        from sklearn.impute import KNNImputer
        transformers.append(('num_knn', Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ]), numerical_features)) # Changed from core_numerical to numerical_features

    # Numerical branch 5: polynomial interactions on core numerical columns only
    if len(core_numerical) > 1 and len(core_numerical) <= 15:
        transformers.append(('num_poly', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
        ]), core_numerical))

    # Categorical branch 1: OHE for low-cardinality categoricals
    if len(cat_low) > 0:
        transformers.append(('cat_ohe', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ]), cat_low))

    # Categorical branch 2: Target Encoder for high-cardinality categoricals
    if len(cat_high) > 0:
        try:
            from sklearn.preprocessing import TargetEncoder
            te = TargetEncoder(target_type='auto', smooth='auto')
            transformers.append(('cat_target', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('te', te)
            ]), cat_high))
        except ImportError:
            pass

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # 3. Model Initialization (tuned LightGBM for small regularized data)
    if task == 'classification':
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        score_func = mutual_info_classif
    else:
        model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        score_func = mutual_info_regression

    # 4. Create Full Pipeline with post-processing feature selection
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('post_imputer', SimpleImputer(strategy='median')),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('select', SelectPercentile(score_func=score_func, percentile=80)),
        ('classifier', model)
    ])
    
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
            score = -np.sqrt(mean_squared_error(y_val, y_pred)) # Negative RMSE
            
        scores.append(score)

    final_score = np.mean(scores)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 6. Generate Submission (if test_path is provided)
    if test_path and Path(test_path).exists():
        print("Generating submission...")
        pipeline.fit(X, y) # Fit full pipeline on all training data
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
        
        # Replicate rank features using columns derived from training set
        for col in all_numerical_cols_after_generic_fe: # Use the full list from training
            if col in test_X.columns and test_X[col].nunique() > 1:
                test_X[f'__rank_{col}'] = test_X[col].rank(method='average', na_option='keep')

        # Replicate pairwise interactions using core_numerical from training to guarantee identical column names
        if 1 < len(core_numerical) <= 15:
            for i in range(len(core_numerical)):
                for j in range(i + 1, len(core_numerical)):
                    c1, c2 = core_numerical[i], core_numerical[j]
                    if c1 in test_X.columns and c2 in test_X.columns: # Ensure columns exist in test
                        test_X[f'__interact_{c1}_x_{c2}'] = test_X[c1] * test_X[c2]
                        test_X[f'__sum_{c1}_{c2}'] = test_X[c1] + test_X[c2]
                        test_X[f'__diff_{c1}_{c2}'] = test_X[c1] - test_X[c2]
                        test_X[f'__ratio_{c1}_div_{c2}'] = test_X[c1] / test_X[c2].replace(0, np.nan)
                        test_X[f'__ratio_{c2}_div_{c1}'] = test_X[c2] / test_X[c1].replace(0, np.nan)
        
        # Apply stored groupby and interaction encodings
        for (cat_col, num_col), grp in groupby_maps.items():
            if cat_col in test_X.columns: # Ensure cat_col exists in test_X
                test_X[f'__grp_mean_{cat_col}_{num_col}'] = test_X[cat_col].map(grp['mean'])
                test_X[f'__grp_std_{cat_col}_{num_col}'] = test_X[cat_col].map(grp['std'])
                test_X[f'__grp_median_{cat_col}_{num_col}'] = test_X[cat_col].map(grp['median'])
            else: # If cat_col missing, fill with a default (e.g., 0 or mean of grp)
                test_X[f'__grp_mean_{cat_col}_{num_col}'] = grp['mean'].mean() # Fallback to global mean
                test_X[f'__grp_std_{cat_col}_{num_col}'] = grp['std'].mean() # Fallback to global mean
                test_X[f'__grp_median_{cat_col}_{num_col}'] = grp['median'].mean() # Fallback to global mean


        for (c1, c2), vc in cat_interact_maps.items():
            if c1 in test_X.columns and c2 in test_X.columns: # Ensure columns exist in test_X
                interact = test_X[c1].astype(str) + '__' + test_X[c2].astype(str)
                test_X[f'__freq_interact_{c1}_{c2}'] = interact.map(vc).fillna(0)
            else: # If any interacting column missing, fill with 0
                 test_X[f'__freq_interact_{c1}_{c2}'] = 0
        
        # Align columns with training (X.columns) and preserve NaNs for proper imputation
        # New columns in test_X (not in X) will be dropped, missing columns (in X but not test_X) will be added as NaN
        test_X = test_X.reindex(columns=X.columns)

        # Align dtypes for categorical columns to avoid sklearn pipeline errors
        for col in X.columns:
            if X[col].dtype == 'object' and col in test_X.columns:
                test_X[col] = test_X[col].astype(object)

        # Sanitize object columns in test_X
        for col in test_X.select_dtypes(include=['object']).columns:
            test_X[col] = test_X[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)

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