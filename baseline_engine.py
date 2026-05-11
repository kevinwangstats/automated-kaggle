import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from logger import log_stage, log_metric, log_error, suppress_stdout_stderr
import os

def create_template_script(dataset_path: str, target_col: str, model_type: str, test_path: str = None, custom_metric: str = None) -> str:
    imports = ""
    model_init = ""
    
    if model_type == "xgb":
        imports = "from xgboost import XGBRegressor, XGBClassifier"
        model_init = f"model = XGBClassifier(random_state=42) if task == 'classification' else XGBRegressor(random_state=42)"
    elif model_type == "lgb":
        imports = "from lightgbm import LGBMRegressor, LGBMClassifier"
        model_init = f"model = LGBMClassifier(random_state=42) if task == 'classification' else LGBMRegressor(random_state=42)"
    elif model_type == "cat":
        imports = "from catboost import CatBoostRegressor, CatBoostClassifier"
        model_init = f"model = CatBoostClassifier(random_state=42, verbose=0) if task == 'classification' else CatBoostRegressor(random_state=42, verbose=0)"
    
    
    # Custom metric handling
    metric_str = f"'{custom_metric}'" if custom_metric else "('roc_auc' if task == 'classification' else 'neg_mean_squared_error')"
    
    inference_block = ""
    if test_path:
        inference_block = f'''
    # 4. Generate Submission
    print("Generating submission...")
    model.fit(X, y)
    test_df = pd.read_csv("{test_path}")
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
    
    preds = model.predict(test_X)
    if task == 'classification':
        preds = le_y.inverse_transform(preds)
        
    submission = pd.DataFrame()
    # Assuming first column of test is ID, or just outputting raw preds
    if len(test_df.columns) > 0:
        submission[test_df.columns[0]] = test_df.iloc[:, 0]
    submission['{target_col}'] = preds
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
'''

    script = f'''import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
{imports}
import re

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv("{dataset_path}")
    target_col = "{target_col}"
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    X.columns = [re.sub(r'[^\\w\\s]', '', col).replace(' ', '_') for col in X.columns]
    
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
    {model_init}
    
    # 3. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {metric_str}
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    final_score = np.mean(scores)
    # We output raw sklearn score so agent loop can always assume higher is better
        
    print(f"FINAL_CV_SCORE: {{final_score:.4f}}")
    {inference_block}
    return final_score

if __name__ == "__main__":
    train_and_evaluate()
'''
    return script


def evaluate_baselines(dataset_path: str, target_col: str, test_path: str = None, custom_metric: str = None):
    log_stage("Baseline Evaluation")
    try:
        df = pd.read_csv(dataset_path)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        df = df.dropna(subset=[target_col])
        y = df[target_col]
        X = df.drop(columns=[target_col])
        import re
        # Clean column names to avoid LightGBM JSON errors
        X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
        
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
        task = 'classification' if y.nunique() < 20 else 'regression'
        if task == 'classification':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
            
        scoring = custom_metric if custom_metric else ('roc_auc' if task == 'classification' else 'neg_mean_squared_error')
        cv = KFold(n_splits=3, shuffle=True, random_state=42) # 3 splits for speed in baseline
        
        results = {}
        metric_reports = {}
        models_to_eval = {}
        
        with suppress_stdout_stderr():
            try:
                from xgboost import XGBRegressor, XGBClassifier
                models_to_eval['xgb'] = XGBClassifier(random_state=42) if task == 'classification' else XGBRegressor(random_state=42)
            except Exception: pass
                
            try:
                from lightgbm import LGBMRegressor, LGBMClassifier
                models_to_eval['lgb'] = LGBMClassifier(random_state=42) if task == 'classification' else LGBMRegressor(random_state=42)
            except Exception: pass
                
            try:
                from catboost import CatBoostRegressor, CatBoostClassifier
                models_to_eval['cat'] = CatBoostClassifier(random_state=42, verbose=0) if task == 'classification' else CatBoostRegressor(random_state=42, verbose=0)
            except Exception: pass

            for m_name, model in models_to_eval.items():
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                    results[m_name] = np.mean(scores)
                    
                    if task == 'classification' and len(np.unique(y)) == 2:
                        from sklearn.model_selection import cross_val_predict
                        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
                        
                        preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
                        acc = accuracy_score(y, preds)
                        f1 = f1_score(y, preds)
                        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        report = f"--- {m_name.upper()} ---\n"
                        report += f"  Pos Cases: {tp+fn} | Neg Cases: {tn+fp}\n"
                        report += f"  Accuracy:    {acc:.4f}\n"
                        report += f"  F1 Score:    {f1:.4f}\n"
                        report += f"  Sensitivity: {sensitivity:.4f}\n"
                        report += f"  Specificity: {specificity:.4f}\n"
                        metric_reports[m_name] = report
                except Exception as e:
                    pass
        
        # Print detailed reports outside the suppression block
        if metric_reports:
            print("\n" + "="*30)
            print("DETAILED BASELINE METRICS (Binary Classification)")
            for r in metric_reports.values():
                print(r)
            print("="*30 + "\n")
                
        if not results:
            raise ValueError("All baseline models failed to evaluate.")
            
        # Determine best (raw sklearn score, so higher is always better)
        best_model = max(results, key=results.get)
        best_score = results[best_model]
            
        log_metric(f"Best Baseline ({best_model})", best_score)
        
        # Generate the script for the best model
        script_content = create_template_script(dataset_path, target_col, best_model, test_path, custom_metric)
        with open("train_model.py", "w") as f:
            f.write(script_content)
            
        return best_score, "train_model.py", task

    except Exception as e:
        log_error("Baseline evaluation failed", e)
        raise
