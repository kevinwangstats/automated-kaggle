import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def feature_engineering(df, age_map=None):
    """
    Applies feature engineering steps.
    If age_map is None, computes median Age per (Title, Pclass) and returns
    a dictionary mapping -> median. Otherwise uses provided map.
    """
    df = df.copy()
    
    # Title extraction from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].apply(lambda x: re.search(r' ([A-Za-z]+)\.', str(x)).group(1) if re.search(r' ([A-Za-z]+)\.', str(x)) else 'Unknown')
        # Group rare titles and unify
        df['Title'] = df['Title'].replace({
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
        })
    
    # Cabin deck
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
    
    # Ticket prefix (first part before space)
    if 'Ticket' in df.columns:
        df['TicketPrefix'] = df['Ticket'].apply(lambda x: str(x).split()[0].replace('.','').replace('/','') if pd.notnull(x) else 'Unknown')
    
    # Family size
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Fare per person
    if 'Fare' in df.columns and 'FamilySize' in df.columns:
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # Age imputation using median per (Title, Pclass)
    if age_map is None and 'Age' in df.columns and 'Title' in df.columns and 'Pclass' in df.columns:
        # Compute median Age per group
        age_map = df.groupby(['Title', 'Pclass'])['Age'].median().to_dict()
        df['Age'] = df.apply(lambda row: age_map.get((row['Title'], row['Pclass']), df['Age'].median()) if pd.isnull(row['Age']) else row['Age'], axis=1)
    elif age_map is not None and 'Age' in df.columns:
        df['Age'] = df.apply(lambda row: age_map.get((row['Title'], row['Pclass']), np.nan) if pd.isnull(row['Age']) else row['Age'], axis=1)
    
    # Fare binning (quantiles)
    if 'Fare' in df.columns:
        df['FareBin'] = pd.qcut(df['Fare'].fillna(-1), 4, labels=False, duplicates='drop').astype(str)
    
    # Drop original columns that have been engineered away
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    if age_map is not None:
        return df, age_map
    return df, age_map

def train_and_evaluate(config_path="config.yaml"):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    df = pd.read_csv(dataset_path, nrows=None)
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    # 2. Feature Engineering (compute age_map from training)
    X, age_map = feature_engineering(X_raw)
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
    
    # Determine task type
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw

    # 3. Identify column types after engineering
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    # 4. Build preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # 5. Hyperparameter tuning for base models
    models_and_params = []
    
    # XGBoost
    try:
        if task == 'classification':
            base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_dist = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'model__gamma': [0, 0.1, 0.2],
                'model__reg_alpha': [0, 0.001, 0.01],
                'model__reg_lambda': [1, 1.5, 2]
            }
        else:
            base = XGBRegressor(random_state=42)
            param_dist = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'model__gamma': [0, 0.1, 0.2],
                'model__reg_alpha': [0, 0.001, 0.01],
                'model__reg_lambda': [1, 1.5, 2]
            }
        models_and_params.append(('xgb', base, param_dist))
    except Exception as e:
        print(f"Could not initialize XGBoost: {e}")
    
    # LightGBM
    try:
        if task == 'classification':
            base = LGBMClassifier(random_state=42, verbose=-1)
            param_dist = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7, -1],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'model__reg_alpha': [0, 0.1, 0.5],
                'model__reg_lambda': [0, 0.1, 0.5]
            }
        else:
            base = LGBMRegressor(random_state=42, verbose=-1)
            param_dist = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7, -1],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'model__reg_alpha': [0, 0.1, 0.5],
                'model__reg_lambda': [0, 0.1, 0.5]
            }
        models_and_params.append(('lgb', base, param_dist))
    except Exception as e:
        print(f"Could not initialize LightGBM: {e}")
    
    # CatBoost
    try:
        if task == 'classification':
            base = CatBoostClassifier(random_state=42, verbose=0)
            param_dist = {
                'model__iterations': [100, 200, 300],
                'model__depth': [4, 6, 8],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__l2_leaf_reg': [1, 3, 5],
                'model__border_count': [32, 64, 128]
            }
        else:
            base = CatBoostRegressor(random_state=42, verbose=0)
            param_dist = {
                'model__iterations': [100, 200, 300],
                'model__depth': [4, 6, 8],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.7, 0.8, 0.9, 1.0],
                'model__l2_leaf_reg': [1, 3, 5],
                'model__border_count': [32, 64, 128]
            }
        models_and_params.append(('cat', base, param_dist))
    except Exception as e:
        print(f"Could not initialize CatBoost: {e}")
    
    # HistGradientBoosting
    try:
        if task == 'classification':
            base = HistGradientBoostingClassifier(random_state=42)
            param_dist = {
                'model__max_iter': [100, 200, 300],
                'model__max_depth': [3, 5, None],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_leaf_nodes': [31, 50, None],
                'model__l2_regularization': [0, 0.1, 0.5]
            }
        else:
            base = HistGradientBoostingRegressor(random_state=42)
            param_dist = {
                'model__max_iter': [100, 200, 300],
                'model__max_depth': [3, 5, None],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_leaf_nodes': [31, 50, None],
                'model__l2_regularization': [0, 0.1, 0.5]
            }
        models_and_params.append(('hist', base, param_dist))
    except Exception as e:
        print(f"Could not initialize HistGradientBoosting: {e}")

    # Tune each model with a quick 3-fold CV on entire dataset (note: mild data leakage, common in practice)
    tuned_estimators = []
    for name, base_model, param_dist in models_and_params:
        print(f"Tuning {name}...")
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', base_model)])
        scoring = 'roc_auc' if task == 'classification' else 'neg_mean_squared_error'
        # Adjusted n_iter to 20 for more thorough tuning
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist,
            n_iter=20, cv=3, scoring=scoring,
            random_state=42, n_jobs=-1, verbose=0
        )
        search.fit(X, y)
        best_est = search.best_estimator_
        tuned_model = best_est.named_steps['model']
        tuned_estimators.append((name, tuned_model))
        print(f"  best score: {search.best_score_:.4f}")

    if not tuned_estimators:
        raise RuntimeError("No models could be initialized.")

    # 6. Build Voting (soft) or Stacking ensemble
    if task == 'classification':
        # Use VotingClassifier with soft voting for probability averaging
        ensemble = VotingClassifier(estimators=tuned_estimators, voting='soft')
    else:
        ensemble = VotingRegressor(estimators=tuned_estimators)

    # Final pipeline: preprocessing + ensemble
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', ensemble)])

    # 7. Evaluate with 5-fold Cross-Validation (stratified for classification)
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'roc_auc'
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'neg_mean_squared_error'
    
    print("Running 5-fold CV...")
    scores = []
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        from sklearn.base import clone
        fold_pipeline = clone(final_pipeline)
        fold_pipeline.fit(X_train, y_train)
        
        if task == 'classification':
            y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
        else:
            y_pred = fold_pipeline.predict(X_val)
            score = -mean_squared_error(y_val, y_pred)  # convert to positive scale
        scores.append(score)

    final_score = np.mean(scores)
    print(f"Final CV score: {final_score:.4f}")
    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 8. Generate submission
    if test_path and os.path.exists(test_path):
        print("Generating submission...")
        final_pipeline.fit(X, y)
        test_df = pd.read_csv(test_path)
        # Apply same feature engineering (with training age_map)
        test_X_raw = test_df.copy()
        test_X, _ = feature_engineering(test_X_raw, age_map=age_map)
        # Align columns with training features
        test_X = test_X.reindex(columns=X.columns, fill_value=np.nan)
        
        if task == 'classification':
            preds = final_pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = final_pipeline.predict(test_X)
            
        submission = pd.DataFrame()
        # Use first column of test set as identifier if available
        if len(test_df.columns) > 0:
            submission[test_df.columns[0]] = test_df.iloc[:, 0]
        submission[target_col] = preds
        submission.to_csv("raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    train_and_evaluate(args.config)