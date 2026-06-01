"""
baseline_engine.py

Generates and evaluates a baseline training script.
Can be executed as an independent module to test baseline generation:
    python baseline_engine.py --config config.yaml --output_dir .workspaces/test
"""
import pandas as pd
import numpy as np
import yaml
import wandb
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, get_scorer, accuracy_score, f1_score, confusion_matrix
from logger import log_stage, log_metric, log_error, log_info, suppress_stdout_stderr
from utils import clean_column_names
import os
import re

def create_template_script(dataset_path: str, target_col: str, best_model_name: str, test_path: str = None, custom_metric: str = None, max_rows: int = None) -> str:
    try:
        with open("models_registry.yaml", "r") as f:
            registry = yaml.safe_load(f)
            if not registry or 'models' not in registry:
                registry = {'models': {}}
    except Exception:
        registry = {'models': {}}

    if best_model_name in registry['models']:
        model_cfg = registry['models'][best_model_name]
        imports_str = model_cfg["imports"]
        init_block = f"if task == 'classification':\n"
        init_block += f"        model = {model_cfg['classifier']}\n"
        init_block += f"    else:\n"
        init_block += f"        model = {model_cfg['regressor']}\n"
    else:
        imports_str = "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor"
        init_block = "if task == 'classification':\n        model = RandomForestClassifier(random_state=42)\n    else:\n        model = RandomForestRegressor(random_state=42)\n"

    metric_str = f"'{custom_metric}'" if custom_metric else "('roc_auc' if task == 'classification' else 'neg_mean_squared_error')"
    nrows_str = f"nrows={max_rows}" if max_rows is not None else "nrows=None"

    script = f'''import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, get_scorer
{imports_str}
from pathlib import Path
from tqdm import tqdm
from utils import load_config, clean_column_names

warnings.filterwarnings('ignore')

def train_and_evaluate(config_path="config.yaml", output_dir="."):
    # 1. Load Configuration & Data
    config = load_config(config_path)
    dataset_path = config.get("dataset_path")
    target_col = config.get("target_col")
    test_path = config.get("test_path")

    df = pd.read_csv(dataset_path, {nrows_str})
    
    # Basic Preprocessing
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    X = clean_column_names(X)
    
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
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ],
        remainder='drop'
    )

    # 3. Model Initialization
    {init_block}

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # 5. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {metric_str}
    
    print(f"Running Cross-Validation (folds=5)...")
    scores = []
    # Manual loop to show progress
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        
        # Scoring
        if task == 'classification':
            if scoring == 'roc_auc':
                if hasattr(fold_pipeline, "predict_proba"):
                    y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
                elif hasattr(fold_pipeline, "decision_function"):
                    y_pred = fold_pipeline.decision_function(X_val)
                else:
                    y_pred = fold_pipeline.predict(X_val)
                score = roc_auc_score(y_val, y_pred)
            else:
                score = get_scorer(scoring)(fold_pipeline, X_val, y_val)
        else:
            score = get_scorer(scoring)(fold_pipeline, X_val, y_val)
            
        scores.append(score)

    final_score = np.mean(scores)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        json.dump({{"cv_score": final_score}}, f)

    # 5.5 Extract Feature Importances (from the last fold)
    try:
        classifier = fold_pipeline.named_steps.get('classifier', fold_pipeline.steps[-1][1])
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            preprocessor = fold_pipeline.named_steps.get('preprocessor')
            if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f"f{{i}}" for i in range(len(importances))]
            
            fi_df = pd.DataFrame({{'feature': feature_names, 'importance': importances}})
            fi_df = fi_df.sort_values('importance', ascending=False)
            fi_data = {{
                'top_15_features': fi_df.head(15)['feature'].tolist(),
                'bottom_15_features': fi_df.tail(15)['feature'].tolist()
            }}
            with open(output_path / "feature_importances.json", "w") as f:
                json.dump(fi_data, f)
    except Exception as e:
        print(f"Could not extract feature importances: {{e}}")

    # 6. Generate Submission (if test_path is provided)
    if test_path and Path(test_path).exists():
        print("Generating submission...")
        pipeline.fit(X, y)
        test_df = pd.read_csv(test_path)
        
        # Ensure test columns match train columns before preprocessing
        test_X = test_df[X.columns.intersection(test_df.columns)]

        if task == 'classification':
            if hasattr(pipeline, "predict_proba"):
                if len(le_y.classes_) > 2:
                    preds = pipeline.predict_proba(test_X)
                else:
                    preds = pipeline.predict_proba(test_X)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                preds = pipeline.decision_function(test_X)
            else:
                preds = pipeline.predict(test_X)
        else:
            preds = pipeline.predict(test_X)
            
        submission = pd.DataFrame()
        if len(test_df.columns) > 0:
             submission[test_df.columns[0]] = test_df.iloc[:, 0]
             
        if task == 'classification' and len(le_y.classes_) > 2 and hasattr(pipeline, "predict_proba"):
            for i in range(preds.shape[1]):
                submission[f"class_{{i}}"] = preds[:, i]
        else:
            submission[target_col] = preds
            
        submission.to_csv(output_path / "raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()
    train_and_evaluate(args.config, args.output_dir)
'''
    return script


def evaluate_baselines(dataset_path: str, target_col: str, test_path: str = None, custom_metric: str = None, wandb_enabled: bool = False, wandb_project: str = None, wandb_entity: str = None, max_rows: int = None, workspace_mgr=None):
    log_stage("Baseline Evaluation")
    try:
        df = pd.read_csv(dataset_path, nrows=max_rows)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        df = df.dropna(subset=[target_col])
        y_raw = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Clean column names to avoid LightGBM JSON errors
        X = clean_column_names(X)
        
        task = 'classification' if y_raw.nunique() < 20 else 'regression'
        if task == 'classification':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y_raw)
        else:
            y = y_raw

        # Define the same preprocessor that will be used in the generated script
        try:
            categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns
        except TypeError:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=np.number).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ],
            remainder='drop'
        )

        scoring = custom_metric if custom_metric else ('roc_auc' if task == 'classification' else 'neg_mean_squared_error')
        cv = KFold(n_splits=3, shuffle=True, random_state=42) # 3 splits for speed in baseline
        
        results = {}
        metric_reports = {}
        models_to_eval = {}
        

        with suppress_stdout_stderr():
            # Initialize models from registry
            try:
                with open("models_registry.yaml", "r") as f:
                    registry = yaml.safe_load(f)
                    if not registry or 'models' not in registry:
                        registry = {'models': {}}
                for m_name, m_config in registry['models'].items():
                    try:
                        exec(m_config["imports"])
                        model_str = m_config["classifier"] if task == 'classification' else m_config["regressor"]
                        models_to_eval[m_name] = eval(model_str)
                    except Exception as e:
                        pass
            except Exception:
                pass

            pbar = tqdm(models_to_eval.items(), desc="Evaluating Baselines")
            for m_name, model in pbar:
                pbar.set_description(f"Evaluating {m_name.upper()}")
                try:
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                    results[m_name] = np.mean(scores)
                    
                    if task == 'classification':
                        preds = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
                        acc = accuracy_score(y, preds)
                        
                        report = f"--- {m_name.upper()} ---\n"
                        
                        if len(np.unique(y)) == 2:
                            f1 = f1_score(y, preds)
                            tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                            
                            report += f"  Pos Cases: {tp+fn} | Neg Cases: {tn+fp}\n"
                            report += f"  Accuracy:    {acc:.4f}\n"
                            report += f"  F1 Score:    {f1:.4f}\n"
                            report += f"  Sensitivity: {sensitivity:.4f}\n"
                            report += f"  Specificity: {specificity:.4f}\n"
                        else:
                            f1_macro = f1_score(y, preds, average='macro')
                            f1_micro = f1_score(y, preds, average='micro')
                            
                            unique, counts = np.unique(y, return_counts=True)
                            class_counts = dict(zip(unique, counts))
                            
                            report += f"  Classes: {len(unique)} | Counts: {class_counts}\n"
                            report += f"  Accuracy:    {acc:.4f}\n"
                            report += f"  Macro F1:    {f1_macro:.4f}\n"
                            report += f"  Micro F1:    {f1_micro:.4f}\n"
                            
                        metric_reports[m_name] = report
                except Exception as e:
                    log_info(f"Failed to evaluate {m_name}: {e}")
                    pass
        
        # Print detailed reports outside the suppression block
        if metric_reports:
            log_info("\n" + "="*30)
            if len(np.unique(y)) == 2:
                log_info("DETAILED BASELINE METRICS (Binary Classification)")
            else:
                log_info("DETAILED BASELINE METRICS (Multi-class Classification)")
            for r in metric_reports.values():
                log_info(r)
            log_info("="*30 + "\n")
                
        if not results:
            raise ValueError("All baseline models failed to evaluate.")
            
        # Determine best (raw sklearn score, so higher is always better)
        best_model = max(results, key=results.get)
        best_score = results[best_model]
            
        log_metric(f"Best Baseline ({best_model})", best_score)

        if wandb_enabled:
            wandb.log({"cv_score": best_score, "best_model": best_model, "iteration": 0})
        
        # Generate the script for the best model
        script_content = create_template_script(dataset_path, target_col, best_model, test_path, custom_metric, max_rows)
        with open("train_model.py", "w") as f:
            f.write(script_content)
        script_path = "train_model.py"
            
        return best_score, script_path, task

    except Exception as e:
        log_error("Baseline evaluation failed", e)
        raise
