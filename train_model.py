import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_max_cardinality=100, grp_min_cat=2, grp_max_cat=50,
                 te_smoothing=10.0, rare_thresh=0.005, log_skew_thresh=1.5):
        self.cat_max_cardinality = cat_max_cardinality
        self.grp_min_cat = grp_min_cat
        self.grp_max_cat = grp_max_cat
        self.te_smoothing = te_smoothing
        self.rare_thresh = rare_thresh
        self.log_skew_thresh = log_skew_thresh
        self.drop_cols = []
        self.text_cols = []
        self.num_cols = []
        self.cat_cols = []
        self.grp_cat_cols = []
        self.freq_maps = {}
        self.count_maps = {}
        self.groupby_stats = {}
        self.target_enc_maps = {}
        self.global_mean = None
        self.rare_maps = {}
        self.log_cols = []
        self.clip_bounds = {}
        self.top_corr_pairs = []
        # New attributes for generic text extraction
        self.abbrev_te_maps = {}
        self.extract_first_num = set()
        self.extract_last_num = set()
        self.extract_abbrev = set()

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        n_rows = len(X_df)

        # Drop constant columns and numeric ID-like columns
        for col in X_df.columns:
            nunique = X_df[col].nunique(dropna=False)
            if nunique == 1:
                self.drop_cols.append(col)
            elif nunique == n_rows and pd.api.types.is_numeric_dtype(X_df[col]):
                self.drop_cols.append(col)

        X_remaining = X_df.drop(columns=self.drop_cols, errors='ignore')

        # Identify column types (robust to pandas version)
        self.text_cols = [
            c for c in X_remaining.columns
            if X_remaining[c].dtype == 'object'
            or str(X_remaining[c].dtype).startswith('category')
            or str(X_remaining[c].dtype) == 'string'
        ]

        self.cat_cols = [
            c for c in self.text_cols
            if X_remaining[c].nunique(dropna=False) <= self.cat_max_cardinality
        ]

        self.num_cols = [
            c for c in X_remaining.columns
            if pd.api.types.is_numeric_dtype(X_remaining[c]) and c not in self.text_cols
        ]

        self.grp_cat_cols = [
            c for c in self.cat_cols
            if self.grp_min_cat <= X_remaining[c].nunique(dropna=False) <= self.grp_max_cat
        ]

        # Limit groupby interactions to prevent OOM
        if len(self.grp_cat_cols) * len(self.num_cols) > 500:
            self.grp_cat_cols = []

        # Rare category grouping for low-cardinality text columns (destined for OHE)
        for col in self.cat_cols:
            counts = X_remaining[col].value_counts(dropna=False)
            min_count = max(1, int(self.rare_thresh * n_rows))
            rare_cats = counts[counts < min_count].index.tolist()
            if rare_cats:
                self.rare_maps[col] = {cat: '__RARE__' for cat in rare_cats}

        # Target encoding for all text columns (fit on train fold only)
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
            self.global_mean = float(np.mean(y_arr))
            for col in self.text_cols:
                temp = pd.DataFrame({'__val__': X_remaining[col], '__y__': y_arr})
                stats = temp.groupby('__val__')['__y__'].agg(['mean', 'count'])
                smoothed = (stats['count'] * stats['mean'] + self.te_smoothing * self.global_mean) / (stats['count'] + self.te_smoothing)
                self.target_enc_maps[col] = smoothed.to_dict()

        # Frequency / count maps
        for col in self.text_cols:
            self.freq_maps[col] = X_remaining[col].value_counts(dropna=False, normalize=True)
            self.count_maps[col] = X_remaining[col].value_counts(dropna=False)

        # Groupby statistics
        for cat in self.grp_cat_cols:
            self.groupby_stats[cat] = {}
            for num in self.num_cols:
                if X_remaining[num].nunique(dropna=False) > 1:
                    grp = X_remaining.groupby(X_remaining[cat].astype(str).fillna('__MISSING__'))[num]
                    self.groupby_stats[cat][num] = grp.agg(['mean', 'std', 'min', 'max'])

        # Numeric transforms: clipping bounds, log transform for skewed non-negative features
        for col in self.num_cols:
            col_min = X_remaining[col].min(skipna=True)
            if pd.notna(col_min) and col_min >= 0:
                skew = X_remaining[col].skew(skipna=True)
                if pd.notna(skew) and skew > self.log_skew_thresh:
                    self.log_cols.append(col)
            low, high = X_remaining[col].quantile([0.01, 0.99])
            self.clip_bounds[col] = (float(low), float(high))

        # Top numeric interactions by correlation with target
        if y is not None and len(self.num_cols) >= 2:
            corr_vals = {}
            y_series = pd.Series(y_arr)
            for col in self.num_cols:
                c = X_remaining[col].corr(y_series)
                if pd.notna(c):
                    corr_vals[col] = abs(c)
            top_k = min(5, len(corr_vals))
            if top_k >= 2:
                top_cols = sorted(corr_vals, key=corr_vals.get, reverse=True)[:top_k]
                for i in range(len(top_cols)):
                    for j in range(i + 1, len(top_cols)):
                        self.top_corr_pairs.append((top_cols[i], top_cols[j]))

        # Generic text extraction flags & abbreviation target encoding
        self.abbrev_te_maps = {}
        self.extract_first_num = set()
        self.extract_last_num = set()
        self.extract_abbrev = set()
        if y is not None:
            for col in self.text_cols:
                s_tmp = X_remaining[col].astype(str).replace('nan', '')
                if s_tmp.str.extract(r'(\d+)')[0].notna().any():
                    self.extract_first_num.add(col)
                if s_tmp.str.extract(r'(\d+)(?!.*\d)')[0].notna().any():
                    self.extract_last_num.add(col)
                abbrev_tmp = s_tmp.str.extract(r'\b([A-Za-z]{2,20})\.')[0]
                if abbrev_tmp.notna().any():
                    self.extract_abbrev.add(col)
                    temp = pd.DataFrame({'__val__': abbrev_tmp, '__y__': y_arr})
                    stats = temp.groupby('__val__')['__y__'].agg(['mean', 'count'])
                    smoothed = (stats['count'] * stats['mean'] + self.te_smoothing * self.global_mean) / (stats['count'] + self.te_smoothing)
                    self.abbrev_te_maps[col] = smoothed.to_dict()

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.drop(columns=self.drop_cols, errors='ignore')

        # Apply rare category mapping before OHE
        for col, mapping in self.rare_maps.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(mapping)

        # Text / categorical features
        for col in self.text_cols:
            if col not in X_df.columns:
                continue
            s = X_df[col].astype(str).replace('nan', '')
            X_df[f"{col}_len"] = s.str.len().astype(float)
            X_df[f"{col}_missing"] = X_df[col].isnull().astype(int)
            X_df[f"{col}_freq"] = X_df[col].map(self.freq_maps.get(col, {})).astype(float)
            X_df[f"{col}_count"] = X_df[col].map(self.count_maps.get(col, {})).astype(float)
            X_df[f"{col}_special"] = s.str.count(r'[^A-Za-z0-9\s]').astype(float)
            X_df[f"{col}_digit"] = s.str.count(r'\d').astype(float)
            X_df[f"{col}_word"] = s.str.split().str.len().astype(float)
            X_df[f"{col}_upper"] = s.str.count(r'[A-Z]').astype(float)
            X_df[f"{col}_title"] = s.str.count(r'\b[A-Z][a-z]*\.').astype(float)
            ll = s.str.extract(r'^([A-Za-z])')[0]
            X_df[f"{col}_leading_letter"] = ll.apply(
                lambda x: ord(x.upper()) - ord('A') if pd.notna(x) and len(str(x)) > 0 else -1
            ).astype(float)
            # Target encoding
            if col in self.target_enc_maps:
                te_map = self.target_enc_maps[col]
                X_df[f"{col}_te"] = X_df[col].map(te_map).fillna(self.global_mean).astype(float)

            # Extracted generic features
            if col in self.extract_first_num:
                X_df[f"{col}_first_num"] = s.str.extract(r'(\d+)')[0].astype(float)
            if col in self.extract_last_num:
                X_df[f"{col}_last_num"] = s.str.extract(r'(\d+)(?!.*\d)')[0].astype(float)
            if col in self.extract_abbrev:
                abbrev = s.str.extract(r'\b([A-Za-z]{2,20})\.')[0]
                X_df[f"{col}_abbrev_te"] = abbrev.map(self.abbrev_te_maps[col]).fillna(self.global_mean).astype(float)

            # Ratios relative to length
            denom = X_df[f"{col}_len"].replace(0, np.nan)
            X_df[f"{col}_digit_ratio"] = X_df[f"{col}_digit"] / denom
            X_df[f"{col}_special_ratio"] = X_df[f"{col}_special"] / denom
            X_df[f"{col}_upper_ratio"] = X_df[f"{col}_upper"] / denom

            # Punctuation indicators
            X_df[f"{col}_has_dot"] = s.str.contains(r'\.', regex=True, na=False).astype(int)
            X_df[f"{col}_has_comma"] = s.str.contains(r',', regex=True, na=False).astype(int)
            X_df[f"{col}_has_paren"] = s.str.contains(r'\(|\)', regex=True, na=False).astype(int)

        # Groupby statistics
        for cat in self.grp_cat_cols:
            if cat not in X_df.columns:
                continue
            grp_key = X_df[cat].astype(str).fillna('__MISSING__')
            for num in self.num_cols:
                if num not in self.groupby_stats.get(cat, {}):
                    continue
                stats = self.groupby_stats[cat][num]
                X_df[f"{cat}_grp_{num}_mean"] = grp_key.map(stats['mean']).astype(float)
                X_df[f"{cat}_grp_{num}_std"] = grp_key.map(stats['std']).astype(float)
                X_df[f"{cat}_grp_{num}_min"] = grp_key.map(stats['min']).astype(float)
                X_df[f"{cat}_grp_{num}_max"] = grp_key.map(stats['max']).astype(float)

        # Numeric features
        for col in self.num_cols:
            if col not in X_df.columns:
                continue
            X_df[f"{col}_missing"] = X_df[col].isnull().astype(int)
            # Clip outliers
            low, high = self.clip_bounds.get(col, (-np.inf, np.inf))
            X_df[col] = X_df[col].clip(lower=low, upper=high)
            # Log transform for skewed non-negative features
            if col in self.log_cols:
                X_df[f"{col}_log1p"] = np.log1p(X_df[col].clip(lower=0))
            X_df[f"{col}_is_zero"] = (X_df[col] == 0).astype(int)
            X_df[f"{col}_is_negative"] = (X_df[col] < 0).astype(int)
            X_df[f"{col}_rank"] = X_df[col].rank(pct=True)

        # Interaction features: product, sum, diff, ratio for top pairs
        for c1, c2 in self.top_corr_pairs:
            if c1 in X_df.columns and c2 in X_df.columns:
                X_df[f"{c1}_x_{c2}"] = X_df[c1] * X_df[c2]
                X_df[f"{c1}_plus_{c2}"] = X_df[c1] + X_df[c2]
                X_df[f"{c1}_minus_{c2}"] = X_df[c1] - X_df[c2]
                denom = X_df[c2].replace(0, np.nan)
                X_df[f"{c1}_div_{c2}"] = X_df[c1] / denom

        # Row-wise aggregates on all numeric columns (original + engineered)
        num_cols_all = list(X_df.select_dtypes(include=np.number).columns)
        if len(num_cols_all) > 0:
            X_df['row_num_mean'] = X_df[num_cols_all].mean(axis=1)
            X_df['row_num_std'] = X_df[num_cols_all].std(axis=1)
            X_df['row_num_max'] = X_df[num_cols_all].max(axis=1)
            X_df['row_num_min'] = X_df[num_cols_all].min(axis=1)
            X_df['row_num_median'] = X_df[num_cols_all].median(axis=1)
            X_df['row_num_sum'] = X_df[num_cols_all].sum(axis=1)
            X_df['row_num_range'] = X_df['row_num_max'] - X_df['row_num_min']
            missing_cols = [c for c in X_df.columns if c.endswith('_missing')]
            if missing_cols:
                X_df['row_missing_total'] = X_df[missing_cols].sum(axis=1)

        return X_df


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd()))
    if config.get("dataset_path") and not Path(config.get("dataset_path")).is_absolute():
        config["dataset_path"] = str(repo_root / config.get("dataset_path"))
    if config.get("test_path") and not Path(config.get("test_path")).is_absolute():
        config["test_path"] = str(repo_root / config.get("test_path"))
    return config


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

    # Probe feature engineer to discover safe columns to drop before preprocessor
    fe_probe = FeatureEngineer()
    fe_probe.fit(X)
    if fe_probe.drop_cols:
        X = X.drop(columns=fe_probe.drop_cols, errors='ignore')

    # Determine original categorical columns before feature engineering
    cat_cols = list(X.select_dtypes(include=['object', 'category', 'string']).columns)
    cat_cols = [c for c in cat_cols if X[c].nunique(dropna=False) <= 100]

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
                ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
            ]), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # 3. Model Initialization (Stacking Ensemble)
    if task == 'classification':
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

        estimators = [
            ('xgb', XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
                gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )),
            ('lgb', LGBMClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.5, reg_lambda=2.0, min_child_samples=10,
                class_weight='balanced',
                random_state=42, verbose=-1, n_jobs=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, border_count=128,
                auto_class_weights='Balanced',
                random_seed=42, verbose=False, thread_count=-1,
                loss_function='Logloss'
            )),
            ('lr', LogisticRegression(
                C=0.5, max_iter=2000, class_weight='balanced',
                random_state=42, n_jobs=1
            ))
        ]
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=0.1, max_iter=10000, class_weight='balanced', random_state=42),
            passthrough=False,
            stack_method='predict_proba',
            cv=5,
            n_jobs=1
        )
        metric = config.get("metric", "roc_auc")
    else:
        estimators = [
            ('xgb', XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
                gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, n_jobs=-1
            )),
            ('lgb', LGBMRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.5, reg_lambda=2.0, min_child_samples=10,
                random_state=42, verbose=-1, n_jobs=-1
            )),
            ('cat', CatBoostRegressor(
                iterations=400, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, border_count=128,
                random_seed=42, verbose=False, thread_count=-1,
                loss_function='RMSE'
            )),
            ('ridge', Ridge(random_state=42))
        ]
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(random_state=42),
            passthrough=False,
            cv=5,
            n_jobs=1
        )
        metric = config.get("metric", "neg_mean_squared_error")

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[
        ('fe', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])
    
    # 5. Cross Validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
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
        if fe_probe.drop_cols:
            test_X = test_X.drop(columns=fe_probe.drop_cols, errors='ignore')
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