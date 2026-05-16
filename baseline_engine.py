import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from logger import log_stage, log_metric, log_error, suppress_stdout_stderr
import os
import re

def create_template_script(dataset_path: str, target_col: str, best_model_name: str, test_path: str = None, custom_metric: str = None, max_rows: int = None) -> str:
    import yaml
    try:
        with open("models_registry.yaml", "r") as f:
            registry = yaml.safe_load(f).get("models", {})
    except Exception:
        registry = {}

    imports_str = "\n".join([m["imports"] for m in registry.values()])
    model_init_classification = "\n".join([f"        try: models.append(('{k}', {v['classifier']}))\n        except Exception: pass" for k, v in registry.items()])
    model_init_regression = "\n".join([f"        try: models.append(('{k}', {v['regressor']}))\n        except Exception: pass" for k, v in registry.items()])

    metric_str = f"'{custom_metric}'" if custom_metric else "('roc_auc' if task == 'classification' else 'neg_mean_squared_error')"
    nrows_str = f"nrows={max_rows}" if max_rows is not None else "nrows=None"

    script = f'''import pandas as pd
import numpy as np
import yaml
import json
import os
import re
import argparse
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.ensemble import VotingClassifier, VotingRegressor
{imports_str}
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_and_evaluate(config_path="config.yaml"):
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
    
    X.columns = [re.sub(r'[^\\w\\s]', '', col).replace(' ', '_') for col in X.columns]
    
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
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # 3. Model Initialization (Multi-Model Ensemble)
    {init_block}
    if not models: raise RuntimeError("No models could be initialized.")
    if task == 'classification':
        ensemble = VotingClassifier(estimators=models, voting='soft')
    else:
        ensemble = VotingRegressor(estimators=models)

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', ensemble)])
    
    # 5. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {metric_str}
    
    print(f"Running Cross-Validation (folds=5)...")
    scores = []
    # Manual loop to show progress
    for train_idx, val_idx in tqdm(list(cv.split(X, y)), desc="CV Progress"):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        
        # Scoring
        if task == 'classification':
            if scoring == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                y_pred = fold_pipeline.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            else:
                from sklearn.metrics import get_scorer
                score = get_scorer(scoring)(fold_pipeline, X_val, y_val)
        else:
            from sklearn.metrics import get_scorer
            score = get_scorer(scoring)(fold_pipeline, X_val, y_val)
            
        scores.append(score)

    final_score = np.mean(scores)
    with open("metrics.json", "w") as f:
        json.dump({{"cv_score": final_score}}, f)

    # 6. Generate Submission (if test_path is provided)
    if test_path and os.path.exists(test_path):
        print("Generating submission...")
        pipeline.fit(X, y)
        test_df = pd.read_csv(test_path)
        
        # Ensure test columns match train columns before preprocessing
        test_X = test_df[X.columns.intersection(test_df.columns)]

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)
            
        submission = pd.DataFrame()
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
'''
    return script


def evaluate_baselines(dataset_path: str, target_col: str, test_path: str = None, custom_metric: str = None, wandb_enabled: bool = False, wandb_project: str = None, wandb_entity: str = None, max_rows: int = None):
    log_stage("Baseline Evaluation")
    try:
        df = pd.read_csv(dataset_path, nrows=max_rows)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        df = df.dropna(subset=[target_col])
        y_raw = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Clean column names to avoid LightGBM JSON errors
        X.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in X.columns]
        
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
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )

        scoring = custom_metric if custom_metric else ('roc_auc' if task == 'classification' else 'neg_mean_squared_error')
        cv = KFold(n_splits=3, shuffle=True, random_state=42) # 3 splits for speed in baseline
        
        results = {}
        metric_reports = {}
        models_to_eval = {}
        
        from sklearn.pipeline import Pipeline
        from tqdm import tqdm

        import yaml
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

            # H2O is kept here for evaluation, but omitted from the generated ensemble script
            try:
                import h2o
                from h2o.sklearn import H2OAutoMLClassifier, H2OAutoMLRegressor
                H2OAutoMLClassifier._estimator_type = "classifier"
                H2OAutoMLRegressor._estimator_type = "regressor"
                h2o.init(verbose=False)
                h2o_model = H2OAutoMLClassifier(max_models=3, seed=42) if task == 'classification' else H2OAutoMLRegressor(max_models=3, seed=42)
                models_to_eval['h2o'] = h2o_model
            except Exception: pass

            pbar = tqdm(models_to_eval.items(), desc="Evaluating Baselines")
            for m_name, model in pbar:
                pbar.set_description(f"Evaluating {m_name.upper()}")
                try:
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
                    n_jobs = 1 if m_name == 'h2o' else -1
                    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
                    results[m_name] = np.mean(scores)
                    
                    if task == 'classification':
                        from sklearn.model_selection import cross_val_predict
                        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
                        
                        preds = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=n_jobs)
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
                    print(f"Failed to evaluate {m_name}: {e}")
                    pass
        
        # Print detailed reports outside the suppression block
        if metric_reports:
            print("\n" + "="*30)
            if len(np.unique(y)) == 2:
                print("DETAILED BASELINE METRICS (Binary Classification)")
            else:
                print("DETAILED BASELINE METRICS (Multi-class Classification)")
            for r in metric_reports.values():
                print(r)
            print("="*30 + "\n")
                
        if not results:
            raise ValueError("All baseline models failed to evaluate.")
            
        # Determine best (raw sklearn score, so higher is always better)
        best_model = max(results, key=results.get)
        best_score = results[best_model]
            
        log_metric(f"Best Baseline ({best_model})", best_score)

        if wandb_enabled:
            import wandb
            wandb.log({"cv_score": best_score, "best_model": best_model, "iteration": 0})
        
        # Generate the script for the best model
        script_content = create_template_script(dataset_path, target_col, best_model, test_path, custom_metric, max_rows)
        with open("train_model.py", "w") as f:
            f.write(script_content)
            
        return best_score, "train_model.py", task

    except Exception as e:
        log_error("Baseline evaluation failed", e)
        raise

