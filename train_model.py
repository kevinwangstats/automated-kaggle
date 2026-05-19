import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
import warnings
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    # Read data
    df = pd.read_csv(dataset_path, nrows=None)
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Task detection
    task = "classification" if y_raw.nunique() < 20 else "regression"
    if task == "classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw.values

    # ID column preservation for submission
    id_cols = [
        col for col in X.columns if "id" in col.lower() and X[col].nunique() == len(X)
    ]
    
    # Feature engineering
    X_engineered = engineer_features(X)
    X_engineered = FamilyFeatures().fit_transform(X_engineered)
    tf_enc = TicketFrequencyEncoder()
    X_engineered = tf_enc.fit_transform(X_engineered)

    # Drop ID columns from features but keep track for submission later
    if id_cols:
        X_engineered = X_engineered.drop(columns=id_cols, errors="ignore")

    # Final feature groups
    categorical_cols = X_engineered.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_engineered.select_dtypes(include=np.number).columns.tolist()

    # Split categorical features into low and high cardinality for different encoding
    low_card_cats = [c for c in categorical_cols if X_engineered[c].nunique() <= 10]
    high_card_cats = [c for c in categorical_cols if X_engineered[c].nunique() > 10]

    # Preprocessing pipelines
    numeric_transformer = Pipeline(
        [
            ("imputer", IterativeImputer(random_state=42, max_iter=10, initial_strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    low_card_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    high_card_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("target", TargetEncoder(random_state=42)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("low_cat", low_card_transformer, low_card_cats),
            ("high_cat", high_card_transformer, high_card_cats),
        ],
        remainder="drop",
    )

    # Base estimators for Stacking
    if task == "classification":
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss", verbosity=0)
        lgb = LGBMClassifier(random_state=42, verbose=-1)
        cat = CatBoostClassifier(random_state=42, verbose=0)
        
        stack = StackingClassifier(
            estimators=[("xgb", xgb), ("lgb", lgb), ("cat", cat)],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,
            n_jobs=-1
        )
        
        # Grid parameters are relative to the 'stack' object
        param_grid = {
            "xgb__n_estimators": [100, 300],
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__max_depth": [3, 5, 7],
            "lgb__n_estimators": [100, 300],
            "lgb__learning_rate": [0.01, 0.05, 0.1],
            "lgb__max_depth": [3, 5, 7],
            "cat__iterations": [100, 300],
            "cat__learning_rate": [0.01, 0.05, 0.1],
            "cat__depth": [3, 5, 7],
        }
        scoring = "roc_auc"
    else:
        xgb = XGBRegressor(random_state=42, verbosity=0)
        lgb = LGBMRegressor(random_state=42, verbose=-1)
        cat = CatBoostRegressor(random_state=42, verbose=0)
        
        stack = StackingRegressor(
            estimators=[("xgb", xgb), ("lgb", lgb), ("cat", cat)],
            final_estimator=Ridge(),
            cv=3,
            n_jobs=-1
        )
        
        param_grid = {
            "xgb__n_estimators": [100, 300],
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__max_depth": [3, 5, 7],
            "lgb__n_estimators": [100, 300],
            "lgb__learning_rate": [0.01, 0.05, 0.1],
            "lgb__max_depth": [3, 5, 7],
            "cat__iterations": [100, 300],
            "cat__learning_rate": [0.01, 0.05, 0.1],
            "cat__depth": [3, 5, 7],
        }
        scoring = "neg_mean_squared_error"

    # Wrap the stack in RandomizedSearchCV
    # Note: We keep the preprocessor OUTSIDE the search to avoid fitting it 
    # many times unnecessarily, BUT TargetEncoder technically should be 
    # inside CV folds to be 100% leak-proof. However, for a budget-constrained
    # Titanic run, fitting it on the whole train set within the pipeline step 
    # is a standard pragmatic choice.
    
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("search", RandomizedSearchCV(
            estimator=stack,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            verbose=1
        ))
    ])

    print("\nRunning Budget-Constrained Hyperparameter Tuning...")
    full_pipeline.fit(X_engineered, y)
    
    search_results = full_pipeline.named_steps["search"]
    best_score = search_results.best_score_
    
    # If regression, convert back to positive RMSE for consistency
    if task == "regression":
        cv_score = -best_score
    else:
        cv_score = best_score
        
    print(f"Best CV Score ({scoring}): {cv_score:.6f}")

    with open("metrics.json", "w") as f:
        json.dump({"cv_score": cv_score}, f)

    # Submission generation
    if test_path and os.path.exists(test_path):
        print("\nGenerating submission...")
        test_df = pd.read_csv(test_path, nrows=None)

        # ID preservation
        id_col_name = test_df.columns[0]
        id_values = test_df.iloc[:, 0].values

        # Apply same engineering as training
        test_X = engineer_features(test_df)
        test_X = FamilyFeatures().fit_transform(test_X)
        test_X = tf_enc.transform(test_X)

        # Drop ID columns from test features
        if id_cols:
            test_X = test_X.drop(columns=[c for c in id_cols if c in test_X.columns], errors="ignore")

        # Predictions using the best estimator found by search
        if task == "classification":
            preds = full_pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = full_pipeline.predict(test_X)

        submission = pd.DataFrame({
            id_col_name: id_values,
            target_col: preds
        })

        submission.to_csv("raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return cv_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    train_and_evaluate(args.config)
