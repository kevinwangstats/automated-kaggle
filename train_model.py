import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
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
import warnings

warnings.filterwarnings('ignore')

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
    
    # Impute missing values using median for numerical and mode for categorical
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.number]:
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])

    # Extract Title from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].astype(str).str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mlle', 'Ms', 'Mme']
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')
        df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        df = df.drop(columns=['Name'])
    
    # Family size / IsAlone
    if all(c in df.columns for c in ['SibSp', 'Parch']):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Ticket related features
    if 'Ticket' in df.columns:
        df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z\./ ]+)', expand=False).fillna('None')
        df['TicketNumber'] = df['Ticket'].str.extract(r'(\d+)$', expand=False).fillna(0).astype(int)
        df = df.drop(columns=['Ticket'])

    # Cabin deck & indicator
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
        df = df.drop(columns=['Cabin'])

    # Feature interactions
    if 'Pclass' in df.columns and 'Fare' in df.columns:
        df['Pclass_Fare'] = df['Pclass'] * df['Fare']

    return df

def train_and_evaluate(config_path="config.yaml", output_dir="."):
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")
    nrows = config.get("nrows", None)

    # Load train
    df = pd.read_csv(dataset_path, nrows=nrows)
    # Ensure target column exists before proceeding
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Clean column names
    X_raw.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X_raw.columns]

    # Detect and drop ID-like column from train (first col if all unique)
    id_col_name = None
    if X_raw.iloc[:, 0].nunique() == len(X_raw):
        id_col_name = X_raw.columns[0]
        X_raw = X_raw.drop(columns=[id_col_name])

    # Feature engineering
    X = engineer_features(X_raw)

    # Task inference
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
        scoring_fn = roc_auc_score
        # Adjust scale_pos_weight for imbalanced classification
        counts = np.bincount(y)
        scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 and counts[1] > 0 else 1.0
    else:
        y = y_raw.values
        scoring_fn = mean_squared_error

    # Preprocessor
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Update preprocessor to include StandardScaler for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', StandardScaler())]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )

    # Model definitions - slightly tuned hyperparameters for better performance
    if task == 'classification':
        estimators = [
            ('xgb', XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.02,
                subsample=0.85, colsample_bytree=0.85, min_child_weight=4,
                gamma=0.25, reg_alpha=0.1, reg_lambda=1.2,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42, n_jobs=-1
            )),
            ('lgb', LGBMClassifier(
                n_estimators=500, max_depth=-1, learning_rate=0.02,
                num_leaves=40, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.2,
                class_weight='balanced',
                random_state=42, verbose=-1, n_jobs=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=500, depth=7, learning_rate=0.02,
                l2_leaf_reg=4.0, border_count=255, random_strength=1.2,
                auto_class_weights='Balanced',
                random_seed=42, verbose=0, thread_count=-1
            )),
            ('hist', HistGradientBoostingClassifier(
                max_iter=500, max_depth=4, learning_rate=0.02,
                l2_regularization=1.2, class_weight='balanced',
                random_state=42
            ))
        ]
        # Using Logistic Regression with L2 regularization for the final estimator
        final_estimator = LogisticRegression(C=0.05, max_iter=1500, solver='liblinear', class_weight='balanced', random_state=42)
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,  # Increased CV folds for stacking
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=False
        )
    else:
        estimators = [
            ('xgb', XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.02,
                subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.2,
                random_state=42, n_jobs=-1
            )),
            ('lgb', LGBMRegressor(
                n_estimators=500, max_depth=-1, learning_rate=0.02,
                num_leaves=40, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.2,
                random_state=42, verbose=-1, n_jobs=-1
            )),
            ('cat', CatBoostRegressor(
                iterations=500, depth=7, learning_rate=0.02,
                l2_leaf_reg=4.0, border_count=255, random_strength=1.2,
                random_seed=42, verbose=0, thread_count=-1
            )),
            ('hist', HistGradientBoostingRegressor(
                max_iter=500, max_depth=4, learning_rate=0.02,
                l2_regularization=1.2,
                random_state=42
            ))
        ]
        # RidgeCV with more folds for the final estimator
        final_estimator = RidgeCV(cv=5)
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5, # Increased CV folds for stacking
            n_jobs=-1,
            passthrough=False
        )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', ensemble)])

    # Cross-validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(tqdm(list(cv.split(X, y)), desc=f"CV Progress (Task: {task})")):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_pipeline = clone(pipeline)
        try:
            fold_pipeline.fit(X_train, y_train)
            if task == 'classification':
                y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
                score = scoring_fn(y_val, y_pred)
            else:
                y_pred = fold_pipeline.predict(X_val)
                score = -scoring_fn(y_val, y_pred)  # negative MSE for consistency with higher-is-better
            scores.append(score)
        except Exception as e:
            print(f"Error during training/evaluation of fold {fold}: {e}")
            # Optionally, you could record a NaN or a very low score for this fold
            scores.append(np.nan)

    # Filter out NaNs before calculating the mean
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        final_score = 0.0 # Or raise an error if no folds were successful
        print("Warning: All CV folds failed. Setting final score to 0.")
    else:
        final_score = float(np.mean(valid_scores))
    
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
        
        # Capture ID column before it might be dropped or transformed
        # Assume the first column is the ID column if no specific ID col from config
        test_id_series = test_df_raw.iloc[:, 0]
        test_id_column_name = test_df_raw.columns[0] # Store name for potential removal

        test_df = test_df_raw.copy()
        test_df.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in test_df.columns]

        # Drop the same ID column name if it exists in the test features after cleaning
        if id_col_name and id_col_name in test_df.columns:
            test_X_raw = test_df.drop(columns=[id_col_name])
        elif test_id_column_name in test_df.columns: # Fallback to the cleaned original first column name
             test_X_raw = test_df.drop(columns=[test_id_column_name])
        else:
             test_X_raw = test_df # Should not happen if test_df_raw has at least one column

        # Drop target column if it accidentally exists in test features
        if target_col in test_X_raw.columns:
            test_X_raw = test_X_raw.drop(columns=[target_col])

        test_X = engineer_features(test_X_raw)

        # Ensure test features match training features, imputing missing columns with 0
        missing_cols = set(X.columns) - set(test_X.columns)
        for c in missing_cols:
            test_X[c] = 0
        test_X = test_X[X.columns] # Ensure the order of columns is the same

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)

        submission = pd.DataFrame()
        # Use the preserved original ID series for submission
        submission[test_id_column_name] = test_id_series.values
        submission[target_col] = preds
        submission.to_csv(os.path.join(output_dir, "raw_submission.csv"), index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_and_evaluate(args.config, args.output_dir)