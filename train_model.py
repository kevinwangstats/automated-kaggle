"""
train_model.py

Improved ensemble training script:
- Dataset-agnostic (config.yaml driven).
- Rich feature engineering (titles, deck, interactions, etc.).
- ColumnTransformer with TargetEncoder (fallback OneHot) + StandardScaler.
- Fixed, well‑performing hyperparameters for base models (avoiding costly grid search).
- Soft voting ensemble without calibration.
- Final cross‑validation score written to metrics.json; submission probabilities to raw_submission.csv.
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    repo_root = os.environ.get("REPO_ROOT", os.getcwd())
    for key in ["dataset_path", "test_path"]:
        if config.get(key) and not os.path.isabs(config.get(key)):
            config[key] = os.path.join(repo_root, config[key])
    return config


def engineer_features(df):
    """
    Fast feature engineering: extracts titles, deck, ticket group size,
    family features, interactions, etc. No slow iterative imputation.
    """
    df = df.copy()

    # Basic imputation
    if "Embarked" in df.columns:
        mode_val = df["Embarked"].mode()
        df["Embarked"] = df["Embarked"].fillna(
            mode_val[0] if not mode_val.empty else "S"
        )
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Title from Name
    if "Name" in df.columns:
        df["NameLength"] = df["Name"].apply(len)
        df["Title"] = (
            df["Name"].astype(str).str.extract(r" ([A-Za-z]+)\.", expand=False)
        )
        title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Don": "Rare",
            "Rev": "Rare",
            "Dr": "Rare",
            "Major": "Rare",
            "Lady": "Rare",
            "Sir": "Rare",
            "Col": "Rare",
            "Capt": "Rare",
            "Countess": "Rare",
            "Jonkheer": "Rare",
            "Dona": "Rare",
        }
        df["Title"] = df["Title"].map(lambda x: title_mapping.get(x, "Rare"))
        df["TitleFreq"] = df.groupby("Title")["Title"].transform("count") / len(df)
        df = df.drop(columns=["Name"])

    # Ticket features
    if "Ticket" in df.columns:
        df["TicketPrefix"] = df["Ticket"].apply(
            lambda x: "".join(re.findall(r"^[A-Za-z]+", str(x)))
            if pd.notna(x)
            else "Unknown"
        )
        df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
        df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]
        df = df.drop(columns=["Ticket"])
    else:
        df["TicketPrefix"] = "Unknown"
        df["TicketGroupSize"] = 1
        df["FarePerPerson"] = df["Fare"] if "Fare" in df.columns else 0

    # Age imputation (fast group median then global median)
    if "Age" in df.columns:
        df["AgeMissing"] = df["Age"].isna().astype(int)
        if (
            "Title" in df.columns
            and "Pclass" in df.columns
            and "Sex" in df.columns
        ):
            df["Age"] = df.groupby(["Pclass", "Sex", "Title"])[
                "Age"
            ].transform(lambda x: x.fillna(x.median()))
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Cabin features
    if "Cabin" in df.columns:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Deck"] = df["Cabin"].apply(
            lambda x: str(x)[0] if pd.notna(x) else "U"
        )
        df["Deck"] = df["Deck"].replace(["T", "A", "G", "F"], "Rare")
        df["DeckFreq"] = df["Deck"].map(df["Deck"].value_counts(normalize=True))
        df = df.drop(columns=["Cabin"])

    # Family/group features
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["GroupSize"] = df[["FamilySize", "TicketGroupSize"]].max(axis=1)
        df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
        df["GroupCategory"] = (
            pd.cut(
                df["GroupSize"],
                bins=[0, 1, 4, 20],
                labels=["Alone", "Small", "Large"],
            )
            .astype(str)
        )
    else:
        df["FamilySize"] = 1
        df["GroupSize"] = 1
        df["IsAlone"] = 1
        df["GroupCategory"] = "Alone"

    # Interactions
    if "Sex" in df.columns and "Pclass" in df.columns:
        df["SexClass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    if "Pclass" in df.columns and "Age" in df.columns:
        df["AgeClass"] = df["Age"] * df["Pclass"]
    if "Sex" in df.columns and "Age" in df.columns:
        bins = [0, 12, 20, 40, 60, 100]
        labels = ["Child", "Teen", "Adult", "Middle", "Senior"]
        age_cat = pd.cut(
            df["Age"], bins=bins, labels=labels, right=False
        ).astype(str)
        df["SexAge"] = df["Sex"].astype(str) + "_" + age_cat

    # Fare features
    if "FarePerPerson" in df.columns:
        df["LogFarePerPerson"] = np.log1p(df["FarePerPerson"])
        lower = df["LogFarePerPerson"].quantile(0.01)
        upper = df["LogFarePerPerson"].quantile(0.99)
        df["LogFarePerPerson"] = df["LogFarePerPerson"].clip(lower, upper)
        df["FareBin"] = (
            pd.qcut(
                df["FarePerPerson"],
                6,
                labels=["VL", "L", "ML", "MH", "H", "VH"],
                duplicates="drop",
            )
            .astype(str)
        )

    if "Age" in df.columns and "FarePerPerson" in df.columns:
        df["Age_mul_FarePerPerson"] = df["Age"] * df["FarePerPerson"]

    if "Age" in df.columns:
        df["AgeSquared"] = df["Age"] ** 2
        df["IsInfant"] = (df["Age"] <= 2).astype(int)
        df["IsElderly"] = (df["Age"] >= 60).astype(int)

    # Drop raw Fare
    if "Fare" in df.columns:
        df = df.drop(columns=["Fare"])

    # Fill any remaining categorical NaNs
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna("Missing").astype(str)

    return df


def train_and_evaluate(config_path="config.yaml", output_dir="."):
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")
    nrows = config.get("nrows", None)

    if not dataset_path or not target_col:
        raise ValueError("dataset_path and target_col must be in config.")

    # ---- Load data ----
    df = pd.read_csv(dataset_path, nrows=nrows)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df_train = df.dropna(subset=[target_col])
    y_raw = df_train[target_col]
    X_raw = df_train.drop(columns=[target_col])

    X_raw.columns = [
        re.sub(r"[^\w\s]", "", col).replace(" ", "_") for col in X_raw.columns
    ]

    # ID column detection
    id_col_name = None
    potential_ids = ["passengerid", "id", "index"]
    first_col = X_raw.columns[0]
    if (
        first_col.lower() in potential_ids
        and X_raw[first_col].nunique() == len(X_raw)
    ):
        id_col_name = first_col
        X_raw = X_raw.drop(columns=[id_col_name])

    # Feature engineering
    X = engineer_features(X_raw)

    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Column types
    categorical_features = list(
        X.select_dtypes(include=["object", "category"]).columns
    )
    numerical_features = list(X.select_dtypes(include=np.number).columns)

    # Preprocessor
    try:
        cat_transformer = TargetEncoder(target_type="binary", random_state=42)
        print("Using TargetEncoder for categorical features.")
    except Exception:
        cat_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        print("Falling back to OneHotEncoder.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # ---- Base models with fixed, well‑tuned hyperparameters ----
    # These avoid expensive grid search while still delivering strong performance.
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    lgb = LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        random_strength=1.0,
        bagging_temperature=0.5,
        random_state=42,
        verbose=0,
    )
    hist = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,
        learning_rate=0.03,
        l2_regularization=1.5,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        min_samples_split=5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    estimators = [
        ("xgb", Pipeline([("preprocessor", preprocessor), ("model", xgb)])),
        ("lgb", Pipeline([("preprocessor", preprocessor), ("model", lgb)])),
        ("cat", Pipeline([("preprocessor", preprocessor), ("model", cat)])),
        ("hist", Pipeline([("preprocessor", preprocessor), ("model", hist)])),
        ("rf", Pipeline([("preprocessor", preprocessor), ("model", rf)])),
    ]

    # ---- Soft voting ensemble ----
    print("Building soft voting ensemble ...")
    voting = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1,
    )

    # ---- Cross‑validation evaluation ----
    print("Evaluating ensemble with 5-fold cross-validation ...")
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        voting, X, y, cv=cv_outer, scoring="roc_auc", n_jobs=-1
    )
    final_score = float(np.mean(scores))
    print(f"Voting Ensemble CV AUC: {final_score:.4f}")

    # ---- Save metrics ----
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)
    print(f"CV Score written to metrics.json: {final_score:.4f}")

    # ---- Generate submission ----
    if test_path and os.path.exists(test_path):
        print("Generating submission ...")
        # Fit the final ensemble on all training data
        voting.fit(X, y)

        # Process test data
        test_df_raw = pd.read_csv(test_path, nrows=nrows)
        test_id_series = test_df_raw.iloc[:, 0].copy()
        test_id_col_name = test_df_raw.columns[0]

        test_df = test_df_raw.copy()
        test_df.columns = [
            re.sub(r"[^\w\s]", "", col).replace(" ", "_")
            for col in test_df.columns
        ]

        if id_col_name and id_col_name in test_df.columns:
            test_X_raw = test_df.drop(columns=[id_col_name])
        elif test_id_col_name in test_df.columns:
            test_X_raw = test_df.drop(columns=[test_id_col_name])
        else:
            test_X_raw = test_df

        if target_col in test_X_raw.columns:
            test_X_raw = test_X_raw.drop(columns=[target_col])

        test_X = engineer_features(test_X_raw)

        # Align columns to training
        for col in set(X.columns) - set(test_X.columns):
            if X[col].dtype in [np.float64, np.int64]:
                test_X[col] = 0
            else:
                test_X[col] = "Missing"
        test_X = test_X[X.columns]

        preds = voting.predict_proba(test_X)[:, 1]

        submission = pd.DataFrame({test_id_col_name: test_id_series.values})
        submission[target_col] = preds
        submission.to_csv(out_dir / "raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_and_evaluate(args.config, args.output_dir)