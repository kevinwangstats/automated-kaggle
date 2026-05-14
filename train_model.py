import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from catboost import CatBoostRegressor, CatBoostClassifier
import re

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv("data/titanic/train.csv")
    target_col = "Survived"
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
    
    # Handle categoricals
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
    task = 'classification' if y.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
    
    # 2. Model Initialization
    model = CatBoostClassifier(random_state=42, verbose=0) if task == 'classification' else CatBoostRegressor(random_state=42, verbose=0)
    
    # 3. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'roc_auc'
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    final_score = np.mean(scores)
    # We output raw sklearn score so agent loop can always assume higher is better
        
    print(f"FINAL_CV_SCORE: {final_score:.4f}")
    
    # 4. Generate Submission
    print("Generating submission...")
    model.fit(X, y)
    test_df = pd.read_csv("data/titanic/test.csv")
    # Simple preprocessing matching train
    test_X = test_df.copy()
    for col in test_X.select_dtypes(include=['object', 'category']).columns:
        test_X[col] = test_X[col].astype(str)
        le = LabelEncoder()
        test_X[col] = le.fit_transform(test_X[col])
        
    # Ensure columns match
    missing_cols = set(X.columns) - set(test_X.columns)
    for c in missing_cols:
        test_X[c] = 0
    test_X = test_X[X.columns]
    
    preds = model.predict_proba(test_X)[:, 1] if task == 'classification' else model.predict(test_X)
        
    submission = pd.DataFrame()
    # Assuming first column of test is ID, or just outputting raw preds
    if len(test_df.columns) > 0:
        submission[test_df.columns[0]] = test_df.iloc[:, 0]
    submission['Survived'] = preds
    submission.to_csv("raw_submission.csv", index=False)
    print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    train_and_evaluate()
