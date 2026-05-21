"""
train_model.py

Dataset‑agnostic training script: loads data, engineers features, evaluates multiple
models through cross‑validation, selects the best one, and generates a submission.
Supports a --config argument (default: config.yaml) and outputs metrics.json and
raw_submission.csv into the directory specified by --output_dir.
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

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Helper: sanitize column names
# ----------------------------------------------------------------------
def sanitize_columns(cols):
    """Replace non‑alphanumeric characters (except underscore) and spaces."""
    return [re.sub(r"[^\w\s]", "", col).replace(" ", "_") for col in cols]


# ----------------------------------------------------------------------
# Config loading (dataset‑agnostic)
# ----------------------------------------------------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    repo_root = os.environ.get("REPO_ROOT", os.getcwd())
    for key in ["dataset_path", "test_path"]:
        if config.get(key) and not os.path.isabs(config.get(key)):
            config[key] = os.path.join(repo_root, config.get(key))
    return config


# ----------------------------------------------------------------------
# Feature engineering (applied to both train and test)
# ----------------------------------------------------------------------
def engineer_features(df):
    """
    Enhanced feature engineering for the Titanic dataset.
    Handles missing values, creates title, deck, ticket, family & interaction features.
    """
    df = df.copy()

    # --- Basic imputation -------------------------------------------------
    if "Embarked" in df.columns:
        mode_val = df["Embarked"].mode()
        df["Embarked"] = df["Embarked"].fillna(
            mode_val[0] if not mode_val.empty else "S"
        )
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # --- Name → Title, NameLength -----------------------------------------
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

    # --- Ticket features --------------------------------------------------
    if "Ticket" in df.columns:
        df["TicketPrefix"] = df["Ticket"].apply(
            lambda x: "".join(re.findall(r"^[A-Za-z]+", str(x)))
            if pd.notna(x)
            else "Unknown"
        )
        df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
        df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]
        # numeric part of ticket
        df["TicketNumber"] = (
            df["Ticket"]
            .astype(str)
            .str.extract(r"(\d+)$")
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        df = df.drop(columns=["Ticket"])
    else:
        df["TicketPrefix"] = "Unknown"
        df["TicketGroupSize"] = 1
        df["FarePerPerson"] = df["Fare"] if "Fare" in df.columns else 0
        df["TicketNumber"] = -1

    # --- Age imputation & derived features --------------------------------
    if "Age" in df.columns:
        df["AgeMissing"] = df["Age"].isna().astype(int)
        if "Title" in df.columns and "Pclass" in df.columns and "Sex" in df.columns:
            df["Age"] = df.groupby(["Pclass", "Sex", "Title"])[
                "Age"
            ].transform(lambda x: x.fillna(x.median()))
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # --- Cabin → HasCabin, Deck, DeckFreq, CabinNumber --------------------
    if "Cabin" in df.columns:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Deck"] = df["Cabin"].apply(
            lambda x: str(x)[0] if pd.notna(x) else "U"
        )
        # Group rare decks
        rare_decks = ["T", "A", "G", "F"]  # seen in EDA
        df["Deck"] = df["Deck"].replace(rare_decks, "Rare")
        df["DeckFreq"] = df["Deck"].map(df["Deck"].value_counts(normalize=True))
        # Numeric part of cabin
        df["CabinNumber"] = (
            df["Cabin"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        df = df.drop(columns=["Cabin"])
    else:
        df["HasCabin"] = 0
        df["Deck"] = "U"
        df["DeckFreq"] = 0
        df["CabinNumber"] = -1

    # --- Family / group features ------------------------------------------
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

    # --- Additional family & demographic flags ----------------------------
    if "Age" in df.columns and "Sex" in df.columns:
        df["IsChild"] = (df["Age"] <= 12).astype(int)
        df["IsWoman"] = (df["Sex"].astype(str) == "female").astype(int)
        df["IsMother"] = (
            (df["Sex"].astype(str) == "female")
            & (df["Parch"] > 0)
            & (df["Age"] > 18)
        ).astype(int)

    # --- Interactions -----------------------------------------------------
    if "Sex" in df.columns and "Pclass" in df.columns:
        df["SexClass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    if "Sex" in df.columns and "Embarked" in df.columns:
        df["SexEmbarked"] = df["Sex"].astype(str) + "_" + df["Embarked"].astype(str)
    if "Pclass" in df.columns and "Age" in df.columns:
        df["AgeClass"] = df["Age"] * df["Pclass"]
    if "Sex" in df.columns and "Age" in df.columns:
        bins = [0, 12, 20, 40, 60, 100]
        labels = ["Child", "Teen", "Adult", "Middle", "Senior"]
        age_cat = pd.cut(
            df["Age"], bins=bins, labels=labels, right=False
        ).astype(str)
        df["SexAge"] = df["Sex"].astype(str) + "_" + age_cat
    if "Deck" in df.columns and "Pclass" in df.columns:
        df["DeckClass"] = df["Deck"].astype(str) + "_" + df["Pclass"].astype(str)

    # --- Fare features ----------------------------------------------------
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
        df["FarePerPersonRank"] = df["FarePerPerson"].rank(pct=True)

    # --- Extra numeric features -------------------------------------------
    if "Age" in df.columns and "FarePerPerson" in df.columns:
        df["Age_mul_FarePerPerson"] = df["Age"] * df["FarePerPerson"]
    if "Age" in df.columns:
        df["AgeSquared"] = df["Age"] ** 2
        df["IsInfant"] = (df["Age"] <= 2).astype(int)
        df["IsElderly"] = (df["Age"] >= 60).astype(int)
    if "FamilySize" in df.columns:
        df["LogFamilySize"] = np.log1p(df["FamilySize"])
    if "Age" in df.columns and "FamilySize" in df.columns:
        df["Age_mul_FamilySize"] = df["Age"] * df["FamilySize"]

    # Drop raw Fare (we have FarePerPerson)
    if "Fare" in df.columns:
        df = df.drop(columns=["Fare"])

    # Fill any remaining categorical NaNs with 'Missing'
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna("Missing").astype(str)

    return df


# ----------------------------------------------------------------------
# Model factory: returns a list of (name, pipeline) candidates
# ----------------------------------------------------------------------
def get_model_candidates(preprocessor, random_state=42):
    """
    Creates a diverse set of classifiers with different hyper‑parameters.
    Each entry is (name, Pipeline([("preprocessor", preprocessor), (name, clf)])).
    """
    models = []

    # --- CatBoost variants -------------------------------------------------
    for lr, depth, iters, l2_leaf, rs in [
        (0.03, 6, 500, 3.0, 1.0),
        (0.02, 5, 700, 5.0, 2.0),
        (0.015, 6, 900, 3.0, 1.0),
        (0.01, 5, 1200, 7.0, 2.0),   # higher iterations, stronger reg
    ]:
        cb = CatBoostClassifier(
            iterations=iters,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2_leaf,
            random_strength=rs,
            bagging_temperature=0.5,
            random_state=random_state,
            verbose=0,
        )
        models.append(("catboost", Pipeline([
            ("preprocessor", preprocessor),
            ("catboost", cb),
        ])))

    # --- XGBoost variants --------------------------------------------------
    for lr, md, n_est, subsample, colsample, alpha, lambd in [
        (0.03, 5, 500, 0.7, 0.7, 0.5, 1.0),
        (0.02, 6, 700, 0.8, 0.8, 0.1, 1.0),
        (0.01, 4, 1000, 0.7, 0.7, 0.5, 1.0),
        (0.015, 5, 1200, 0.8, 0.8, 0.2, 1.5),
    ]:
        xgb = XGBClassifier(
            n_estimators=n_est,
            max_depth=md,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample,
            reg_alpha=alpha,
            reg_lambda=lambd,
            random_state=random_state,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        models.append(("xgb", Pipeline([
            ("preprocessor", preprocessor),
            ("xgb", xgb),
        ])))

    # --- LightGBM variants -------------------------------------------------
    for lr, n_est, num_leaves, subsample, colsample in [
        (0.03, 500, 31, 0.7, 0.7),
        (0.02, 700, 50, 0.8, 0.8),
        (0.015, 900, 63, 0.8, 0.8),
    ]:
        lgb = LGBMClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbose=-1,
        )
        models.append(("lgb", Pipeline([
            ("preprocessor", preprocessor),
            ("lgb", lgb),
        ])))

    # --- HistGradientBoosting ----------------------------------------------
    for lr, mi, md, l2 in [
        (0.03, 500, 5, 0.5),
        (0.02, 800, 4, 1.0),
        (0.015, 1000, 6, 2.0),
    ]:
        hist = HistGradientBoostingClassifier(
            max_iter=mi,
            learning_rate=lr,
            max_depth=md,
            l2_regularization=l2,
            random_state=random_state,
        )
        models.append(("histgb", Pipeline([
            ("preprocessor", preprocessor),
            ("histgb", hist),
        ])))

    # --- RandomForest ------------------------------------------------------
    for n_est, max_depth, min_samples_leaf in [
        (500, 8, 2),
        (700, None, 4),
        (1000, 10, 1),
    ]:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        models.append(("rf", Pipeline([
            ("preprocessor", preprocessor),
            ("rf", rf),
        ])))

    # --- ExtraTrees --------------------------------------------------------
    et = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    models.append(("extra_trees", Pipeline([
        ("preprocessor", preprocessor),
        ("extra_trees", et),
    ])))

    # --- GradientBoosting (sklearn) ----------------------------------------
    for lr, n_est, md in [
        (0.03, 500, 4),
        (0.02, 700, 5),
    ]:
        gb = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=md,
            subsample=0.8,
            random_state=random_state,
        )
        models.append(("gb", Pipeline([
            ("preprocessor", preprocessor),
            ("gb", gb),
        ])))

    # --- VotingClassifier (soft) combinations ------------------------------
    voting_configs = [
        {
            "name": "voting_1",
            "estimators": [
                ("xgb", XGBClassifier(n_estimators=700, max_depth=5, learning_rate=0.02,
                                      subsample=0.8, colsample_bytree=0.8,
                                      reg_alpha=0.5, reg_lambda=1.0,
                                      random_state=random_state, verbosity=0,
                                      use_label_encoder=False, eval_metric="logloss")),
                ("lgb", LGBMClassifier(n_estimators=700, learning_rate=0.02, num_leaves=31,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=random_state, verbose=-1)),
                ("cat", CatBoostClassifier(iterations=700, depth=6, learning_rate=0.02,
                                           l2_leaf_reg=3.0, random_strength=1.0,
                                           bagging_temperature=0.5,
                                           random_state=random_state, verbose=0)),
                ("hist", HistGradientBoostingClassifier(max_iter=700, learning_rate=0.02,
                                                        max_depth=5,
                                                        random_state=random_state)),
            ],
            "voting": "soft",
        },
        {
            "name": "voting_2",
            "estimators": [
                ("xgb", XGBClassifier(n_estimators=700, max_depth=5, learning_rate=0.02,
                                      subsample=0.8, colsample_bytree=0.8,
                                      random_state=random_state, verbosity=0,
                                      use_label_encoder=False, eval_metric="logloss")),
                ("lgb", LGBMClassifier(n_estimators=700, learning_rate=0.02, num_leaves=31,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=random_state, verbose=-1)),
                ("cat", CatBoostClassifier(iterations=700, depth=6, learning_rate=0.02,
                                           l2_leaf_reg=3.0, random_state=random_state,
                                           verbose=0)),
            ],
            "voting": "soft",
        },
    ]
    for cfg in voting_configs:
        vc = VotingClassifier(
            estimators=cfg["estimators"], voting=cfg["voting"], n_jobs=-1
        )
        models.append((cfg["name"], Pipeline([
            ("preprocessor", preprocessor),
            ("voting", vc),
        ])))

    # --- StackingClassifier ------------------------------------------------
    # Combine several strong base models with a LogisticRegression meta-learner
    base_estimators = [
        ("cat", CatBoostClassifier(iterations=700, depth=6, learning_rate=0.02,
                                   l2_leaf_reg=3.0, random_strength=1.0,
                                   bagging_temperature=0.5,
                                   random_state=random_state, verbose=0)),
        ("xgb", XGBClassifier(n_estimators=700, max_depth=5, learning_rate=0.02,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.5, reg_lambda=1.0,
                              random_state=random_state, verbosity=0,
                              use_label_encoder=False, eval_metric="logloss")),
        ("lgb", LGBMClassifier(n_estimators=700, learning_rate=0.02, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=random_state, verbose=-1)),
        ("hist", HistGradientBoostingClassifier(max_iter=700, learning_rate=0.02,
                                                max_depth=5,
                                                random_state=random_state)),
        ("rf", RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=2,
                                      class_weight="balanced", random_state=random_state,
                                      n_jobs=-1)),
    ]
    # Two stacking variants with different meta‑learner regularisation
    for C_val in [0.1, 1.0]:
        meta = LogisticRegression(C=C_val, solver="lbfgs", max_iter=1000, random_state=random_state)
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta,
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        models.append((f"stacking_{C_val}", Pipeline([
            ("preprocessor", preprocessor),
            ("stacking", stack),
        ])))

    return models


# ----------------------------------------------------------------------
# Main training & evaluation
# ----------------------------------------------------------------------
def train_and_evaluate(config_path="config.yaml", output_dir="."):
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")
    nrows = config.get("nrows", None)

    if not dataset_path or not target_col:
        raise ValueError("dataset_path and target_col must be in config.")

    # ---- Load data --------------------------------------------------------
    df = pd.read_csv(dataset_path, nrows=nrows)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df_train = df.dropna(subset=[target_col])
    y_raw = df_train[target_col]
    X_raw = df_train.drop(columns=[target_col])

    # sanitize column names
    X_raw.columns = sanitize_columns(X_raw.columns)

    # ---- ID column detection & removal -----------------------------------
    id_col_name = None
    potential_ids = ["passengerid", "id", "index"]
    first_col = X_raw.columns[0]
    if (
        first_col.lower() in potential_ids
        and X_raw[first_col].nunique() == len(X_raw)
    ):
        id_col_name = first_col
        X_raw = X_raw.drop(columns=[id_col_name])

    # ---- Feature engineering ----------------------------------------------
    X = engineer_features(X_raw)
    X.columns = sanitize_columns(X.columns)

    # ---- Target encoding --------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # ---- Column categorisation & preprocessor ----------------------------
    categorical_features = list(
        X.select_dtypes(include=["object", "category"]).columns
    )
    numerical_features = list(X.select_dtypes(include=np.number).columns)

    # Prefer TargetEncoder, fallback to OneHotEncoder
    try:
        cat_transformer = TargetEncoder(target_type="binary", random_state=42, smooth=10)
        print("Using TargetEncoder for categorical features (smooth=10).")
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

    # ---- Candidate models -------------------------------------------------
    print("Assembling candidate models ...")
    candidates = get_model_candidates(preprocessor, random_state=42)

    # ---- Evaluate all candidates via 10‑fold CV ----------------------------
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_model = None
    best_score = -1.0
    best_name = ""

    for name, pipeline in candidates:
        try:
            scores = cross_val_score(
                pipeline, X, y, cv=outer_cv, scoring="roc_auc", n_jobs=-1
            )
            mean_score = float(np.mean(scores))
            print(f"Model {name:20s}  ->  CV AUC = {mean_score:.6f}")
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_model = pipeline
        except Exception as e:
            print(f"Model {name} failed: {e}")

    print(f"\nBest model: {best_name} with CV AUC = {best_score:.6f}")

    # ---- Save metrics -----------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"cv_score": best_score}, f)
    print(f"CV Score written to metrics.json: {best_score:.6f}")

    # ---- Generate submission ----------------------------------------------
    if test_path and os.path.exists(test_path) and best_model is not None:
        print("Generating submission with best model ...")
        # Refit best pipeline on full training data
        best_model.fit(X, y)

        # Process test set
        test_df_raw = pd.read_csv(test_path, nrows=nrows)
        test_id_series = test_df_raw.iloc[:, 0].copy()
        test_id_col_name = test_df_raw.columns[0]

        test_df = test_df_raw.copy()
        test_df.columns = sanitize_columns(test_df.columns)

        # Remove id column from test features
        if id_col_name and id_col_name in test_df.columns:
            test_X_raw = test_df.drop(columns=[id_col_name])
        elif test_id_col_name in test_df.columns:
            test_X_raw = test_df.drop(columns=[test_id_col_name])
        else:
            test_X_raw = test_df

        if target_col in test_X_raw.columns:
            test_X_raw = test_X_raw.drop(columns=[target_col])

        test_X = engineer_features(test_X_raw)
        test_X.columns = sanitize_columns(test_X.columns)

        # Align columns with training set
        for col in set(X.columns) - set(test_X.columns):
            if X[col].dtype in [np.float64, np.int64]:
                test_X[col] = 0
            else:
                test_X[col] = "Missing"
        test_X = test_X[X.columns]

        preds = best_model.predict_proba(test_X)[:, 1]

        submission = pd.DataFrame({test_id_col_name: test_id_series.values})
        submission[target_col] = preds
        submission.to_csv(out_dir / "raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return best_score


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save outputs (metrics.json, raw_submission.csv)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_and_evaluate(args.config, args.output_dir)