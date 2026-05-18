import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
import warnings
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Groups infrequent categories (count < min_freq) into 'Other'."""

    def __init__(self, min_freq=10):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.frequent_cats_ = {}
        for col in X.columns:
            vc = X[col].value_counts()
            self.frequent_cats_[col] = vc[vc >= self.min_freq].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.frequent_cats_:
                allowed = self.frequent_cats_[col]
                X[col] = X[col].apply(lambda x: x if x in allowed else "Other")
        return X


class FamilyFeatures(BaseEstimator, TransformerMixin):
    """Create family size and is_alone from SibSp and Parch."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if "SibSp" in X.columns and "Parch" in X.columns:
            X["family_size"] = X["SibSp"] + X["Parch"] + 1
            X["is_alone"] = (X["family_size"] == 1).astype(int)
        return X


class TicketFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Computes ticket frequency from the training set and maps it to both
    train and test. Also calculates fare_per_person when Fare exists.
    """

    def __init__(self):
        self.ticket_freq_map_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if "Ticket" in X.columns:
            ticket_col = X["Ticket"].astype(str)
            self.ticket_freq_map_ = ticket_col.value_counts().to_dict()
        else:
            self.ticket_freq_map_ = {}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if "Ticket" in X.columns and self.ticket_freq_map_ is not None:
            X["ticket_freq"] = (
                X["Ticket"]
                .astype(str)
                .map(self.ticket_freq_map_)
                .fillna(1)
                .astype(int)
            )
            if "Fare" in X.columns:
                X["fare_per_person"] = X["Fare"] / X["ticket_freq"].clip(lower=1)
        return X


def engineer_features(df):
    """Creates generic features from string columns: length, word count, title, prefix, number."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].astype(str)
        # Basic string features
        df[col + "_len"] = s.str.len()
        df[col + "_wordcnt"] = s.str.count(" ") + 1

        # Extract title pattern: word preceding a period after a comma, e.g., "Mr.", "Mrs."
        title = s.str.extract(r",\s*(\w+)\.", expand=False)
        df[col + "_title"] = title.fillna("Unknown")

        # Extract prefix (leading letters before a digit), e.g., "C" from "C85"
        prefix = s.str.extract(r"^([A-Za-z]+)", expand=False)
        df[col + "_prefix"] = prefix.fillna("Unknown")

        # Extract numeric part
        numbers = s.str.extract(r"(\d+)", expand=False)
        df[col + "_number"] = pd.to_numeric(numbers, errors="coerce")
    return df


def train_and_evaluate(config_path="config.yaml"):
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    # Read data, preserving nrows argument (None = all rows, prevents OOM)
    df = pd.read_csv(dataset_path, nrows=None)

    # Basic cleanup: drop rows with missing target
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Sanitize column names
    X.columns = [re.sub(r"[^\w\s]", "", col).replace(" ", "_") for col in X.columns]

    # Task detection
    task = "classification" if y_raw.nunique() < 20 else "regression"
    if task == "classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # Drop ID-like columns (name contains 'id' and all values are unique)
    id_cols = [
        col for col in X.columns if "id" in col.lower() and X[col].nunique() == len(X)
    ]
    if id_cols:
        X = X.drop(columns=id_cols, errors="ignore")
        print(f"Dropped ID-like columns: {id_cols}")

    # Feature engineering (creates new features from existing object columns)
    X = engineer_features(X)

    # Add family and ticket related features
    X = FamilyFeatures().fit_transform(X)
    tf_enc = TicketFrequencyEncoder()
    X = tf_enc.fit_transform(X)

    # Ensure all object columns are strings (avoids mixed bool/str in OneHotEncoder)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)

    # Drop columns with very high missing rate (>85%) or constant/all‑na
    drop_cols = []
    for col in X.columns:
        miss_ratio = X[col].isna().mean()
        if (
            miss_ratio > 0.85
            or X[col].isna().all()
            or X[col].nunique(dropna=False) <= 1
        ):
            drop_cols.append(col)
    if drop_cols:
        print(f"Dropping columns due to missing/constant: {drop_cols}")
    X = X.drop(columns=drop_cols, errors="ignore")

    # Drop original high-cardinality object columns (keep extracted features)
    high_card_cols = [
        col
        for col in X.select_dtypes(include=["object"]).columns
        if X[col].nunique() > 100
    ]
    if high_card_cols:
        print(f"Dropping high-cardinality columns: {high_card_cols}")
    X = X.drop(columns=high_card_cols, errors="ignore")

    # Final feature groups
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    random_state=42, max_iter=10, initial_strategy="median"
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="MISSING"),
            ),
            ("rare_grouper", RareCategoryGrouper(min_freq=10)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # ------------------------------------------------------------------------
    # Base models – diverse set with tuned hyperparameters
    # ------------------------------------------------------------------------
    models = []

    # XGBoost
    try:
        if task == "classification":
            xgb = XGBClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            xgb = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
        models.append(("xgb", xgb))
    except Exception as e:
        print(f"XGBoost not available: {e}")

    # LightGBM
    try:
        if task == "classification":
            lgb = LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                force_row_wise=True,
            )
        else:
            lgb = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                force_row_wise=True,
            )
        models.append(("lgb", lgb))
    except Exception as e:
        print(f"LightGBM not available: {e}")

    # CatBoost
    try:
        if task == "classification":
            cat = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=4,
                l2_leaf_reg=5,
                random_seed=42,
                verbose=0,
            )
        else:
            cat = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=4,
                l2_leaf_reg=5,
                random_seed=42,
                verbose=0,
            )
        models.append(("cat", cat))
    except Exception as e:
        print(f"CatBoost not available: {e}")

    # HistGradientBoosting
    try:
        if task == "classification":
            hist = HistGradientBoostingClassifier(
                max_iter=1000,
                learning_rate=0.05,
                max_depth=4,
                l2_regularization=0.1,
                random_state=42,
            )
        else:
            hist = HistGradientBoostingRegressor(
                max_iter=1000,
                learning_rate=0.05,
                max_depth=4,
                l2_regularization=0.1,
                random_state=42,
            )
        models.append(("hist", hist))
    except Exception as e:
        print(f"HistGradientBoosting not available: {e}")

    # Random Forest & Extra Trees
    try:
        if task == "classification":
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            et = ExtraTreesClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            et = ExtraTreesRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        models.append(("rf", rf))
        models.append(("et", et))
    except Exception as e:
        print(f"Random Forest / ExtraTrees not available: {e}")

    # SVM with RBF kernel
    try:
        if task == "classification":
            svm = SVC(kernel="rbf", probability=True, random_state=42)
        else:
            svm = SVR(kernel="rbf")
        models.append(("svm", svm))
    except Exception as e:
        print(f"SVM not available: {e}")

    # Logistic Regression (strong linear baseline)
    try:
        if task == "classification":
            lr = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=42
            )
        else:
            lr = None
        if lr:
            models.append(("lr", lr))
    except Exception as e:
        print(f"LogisticRegression not available: {e}")

    # KNN (distance‑based, may help with local patterns)
    try:
        if task == "classification":
            knn = KNeighborsClassifier(n_neighbors=11, weights="distance")
        else:
            knn = KNeighborsRegressor(n_neighbors=11, weights="distance")
        models.append(("knn", knn))
    except Exception as e:
        print(f"KNN not available: {e}")

    if not models:
        raise RuntimeError("No models could be initialized.")

    # ------------------------------------------------------------------------
    # Stacking ensemble with a regularized meta‑learner
    # ------------------------------------------------------------------------
    if task == "classification":
        final_estimator = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=42
        )
        ensemble = StackingClassifier(
            estimators=models,
            final_estimator=final_estimator,
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
        )
    else:
        final_estimator = Ridge(alpha=1.0, random_state=42)
        ensemble = StackingRegressor(
            estimators=models,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,
        )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("estimator", ensemble)]
    )

    # ------------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------------
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    print("\nRunning 5‑fold CV...")
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train_fold, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val = y[train_idx], y[val_idx]

        from sklearn.base import clone

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train_fold, y_train_fold)

        if task == "classification":
            y_pred_proba = fold_pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
        else:
            y_pred = fold_pipeline.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            score = -rmse  # higher = better
        scores.append(score)

    final_score = np.mean(scores)
    print(f"CV Score: {final_score:.6f}")

    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # ------------------------------------------------------------------------
    # Submission generation
    # ------------------------------------------------------------------------
    if test_path and os.path.exists(test_path):
        print("\nGenerating submission...")
        pipeline.fit(X, y)

        test_df = pd.read_csv(test_path, nrows=None)

        # Drop same ID‑like columns
        if id_cols:
            test_df = test_df.drop(
                columns=[c for c in id_cols if c in test_df.columns], errors="ignore"
            )

        # Apply same engineering and transformations as training
        test_X = engineer_features(test_df)
        test_X = FamilyFeatures().fit_transform(test_X)
        test_X = tf_enc.transform(test_X)  # use fitted ticket frequencies

        for col in test_X.select_dtypes(include=["object"]).columns:
            test_X[col] = test_X[col].astype(str)

        test_X = test_X.drop(
            columns=[c for c in drop_cols if c in test_X.columns], errors="ignore"
        )
        test_X = test_X.drop(
            columns=[c for c in high_card_cols if c in test_X.columns], errors="ignore"
        )

        # Align columns with training (only common ones)
        common_cols = [c for c in X.columns if c in test_X.columns]
        if len(common_cols) < len(X.columns):
            missing = set(X.columns) - set(common_cols)
            print(f"Warning: Test data missing columns: {missing}")
        test_X = test_X[common_cols].copy()

        if task == "classification":
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        submission = pd.DataFrame()
        if len(test_df.columns) > 0:
            submission[test_df.columns[0]] = test_df.iloc[:, 0]
        else:
            submission["id"] = np.arange(len(preds))
        submission[target_col] = preds

        submission.to_csv("raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    train_and_evaluate(args.config)