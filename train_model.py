import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import re

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv("data/titanic/train.csv")
    target_col = "Survived"
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
    
    task = 'classification' if y_raw.nunique() < 20 else 'regression'
    if task == 'classification':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
    else:
        y = y_raw

    # 2. Define Preprocessing Pipeline
    try:
        categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns
    except TypeError:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # 3. Model Initialization (Multi-Model Ensemble)
    models = []
    if task == 'classification':
        try: models.append(('xgb', XGBClassifier(random_state=42)))
        except NameError: pass
        try: models.append(('lgb', LGBMClassifier(random_state=42, verbose=-1)))
        except NameError: pass
        try: models.append(('cat', CatBoostClassifier(random_state=42, verbose=0)))
        except NameError: pass
        
        if not models: raise RuntimeError("No models could be initialized.")
        ensemble = VotingClassifier(estimators=models, voting='soft')
    else:
        try: models.append(('xgb', XGBRegressor(random_state=42)))
        except NameError: pass
        try: models.append(('lgb', LGBMRegressor(random_state=42, verbose=-1)))
        except NameError: pass
        try: models.append(('cat', CatBoostRegressor(random_state=42, verbose=0)))
        except NameError: pass
        
        if not models: raise RuntimeError("No models could be initialized.")
        ensemble = VotingRegressor(estimators=models)

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', ensemble)])
    
    # 5. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'roc_auc'
    
    # Using n_jobs=1 because H2O and CatBoost have internal parallelism
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)

    final_score = np.mean(scores)
    import json
    with open("metrics.json", "w") as f:
        json.dump({"cv_score": final_score}, f)

    # 6. Generate Submission (if test_path is provided)

    if "data/titanic/test.csv":
        print("Generating submission...")
        pipeline.fit(X, y)
        test_df = pd.read_csv("data/titanic/test.csv")
        
        # Ensure test columns match train columns before preprocessing
        test_X = test_df[X.columns.intersection(test_df.columns)]

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)
            
        submission = pd.DataFrame()
        if len(test_df.columns) > 0 and test_df.columns[0] in test_df:
             submission[test_df.columns[0]] = test_df.iloc[:, 0]
        submission['Survived'] = preds
        submission.to_csv("raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    train_and_evaluate()
