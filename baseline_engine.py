import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from logger import log_stage, log_metric, log_error, suppress_stdout_stderr
import os
import re

def create_template_script(dataset_path: str, target_col: str, best_model_name: str, test_path: str = None, custom_metric: str = None) -> str:
    # This function now generates a script that uses a robust scikit-learn Pipeline.
    # It solves data leakage and brittle-preprocessing issues.
    
    metric_str = f"'{custom_metric}'" if custom_metric else "('roc_auc' if task == 'classification' else 'neg_mean_squared_error')"

    script = f'''import pandas as pd
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
import h2o
from h2o.sklearn import H2OAutoMLClassifier, H2OAutoMLRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
import re

class H2OClassifier(H2OAutoMLClassifier, ClassifierMixin):
    _estimator_type = "classifier"

class H2ORegressor(H2OAutoMLRegressor, RegressorMixin):
    _estimator_type = "regressor"

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv("{dataset_path}")
    target_col = "{target_col}"
    
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
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # 3. Model Initialization (Multi-Model)
    try:
        h2o.init(verbose=False)
    except Exception:
        pass
    
    models = []
    if task == 'classification':
        try: models.append(('xgb', XGBClassifier(random_state=42)))
        except NameError: pass
        try: models.append(('lgb', LGBMClassifier(random_state=42, verbose=-1)))
        except NameError: pass
        try: models.append(('cat', CatBoostClassifier(random_state=42, verbose=0)))
        except NameError: pass
        try: models.append(('h2o', H2OClassifier(max_models=3, seed=42)))
        except (NameError, Exception): pass
        
        if not models: raise RuntimeError("No models could be initialized.")
        ensemble = VotingClassifier(estimators=models, voting='soft')
    else:
        try: models.append(('xgb', XGBRegressor(random_state=42)))
        except NameError: pass
        try: models.append(('lgb', LGBMRegressor(random_state=42, verbose=-1)))
        except NameError: pass
        try: models.append(('cat', CatBoostRegressor(random_state=42, verbose=0)))
        except NameError: pass
        try: models.append(('h2o', H2ORegressor(max_models=3, seed=42)))
        except (NameError, Exception): pass
        
        if not models: raise RuntimeError("No models could be initialized.")
        ensemble = VotingRegressor(estimators=models)

    # 4. Create Full Pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', ensemble)])
    
    # 5. Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {metric_str}
    
    # Using n_jobs=1 because H2O and CatBoost have internal parallelism
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
    
    final_score = np.mean(scores)
    print(f"FINAL_CV_SCORE: {{final_score:.4f}}")

    # 6. Generate Submission (if test_path is provided)
    if "{test_path}":
        print("Generating submission...")
        pipeline.fit(X, y)
        test_df = pd.read_csv("{test_path}")
        
        # Ensure test columns match train columns before preprocessing
        test_X = test_df[X.columns.intersection(test_df.columns)]

        if task == 'classification':
            preds = pipeline.predict_proba(test_X)[:, 1]
        else:
            preds = pipeline.predict(test_X)
            
        submission = pd.DataFrame()
        if len(test_df.columns) > 0 and test_df.columns[0] in test_df:
             submission[test_df.columns[0]] = test_df.iloc[:, 0]
        submission['{target_col}'] = preds
        submission.to_csv("raw_submission.csv", index=False)
        print("Saved raw_submission.csv")

    return final_score

if __name__ == "__main__":
    train_and_evaluate()
'''
    return script


def evaluate_baselines(dataset_path: str, target_col: str, test_path: str = None, custom_metric: str = None, wandb_enabled: bool = False, wandb_project: str = None, wandb_entity: str = None):
    log_stage("Baseline Evaluation")
    try:
        df = pd.read_csv(dataset_path)
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
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        scoring = custom_metric if custom_metric else ('roc_auc' if task == 'classification' else 'neg_mean_squared_error')
        cv = KFold(n_splits=3, shuffle=True, random_state=42) # 3 splits for speed in baseline
        
        results = {}
        metric_reports = {}
        models_to_eval = {}
        
        from sklearn.pipeline import Pipeline

        with suppress_stdout_stderr():
            # Initialize models
            try:
                from xgboost import XGBRegressor, XGBClassifier
                models_to_eval['xgb'] = XGBClassifier(random_state=42) if task == 'classification' else XGBRegressor(random_state=42)
            except Exception: pass
            try:
                from lightgbm import LGBMRegressor, LGBMClassifier
                models_to_eval['lgb'] = LGBMClassifier(random_state=42, verbose=-1) if task == 'classification' else LGBMRegressor(random_state=42, verbose=-1)
            except Exception: pass
            try:
                from catboost import CatBoostRegressor, CatBoostClassifier
                models_to_eval['cat'] = CatBoostClassifier(random_state=42, verbose=0) if task == 'classification' else CatBoostRegressor(random_state=42, verbose=0)
            except Exception: pass
            try:
                import h2o
                from h2o.sklearn import H2OAutoMLClassifier, H2OAutoMLRegressor
                from sklearn.base import ClassifierMixin, RegressorMixin
                h2o.init(verbose=False)
                class H2OClf(H2OAutoMLClassifier, ClassifierMixin): _estimator_type = "classifier"
                class H2OReg(H2OAutoMLRegressor, RegressorMixin): _estimator_type = "regressor"
                h2o_model = H2OClf(max_models=3, seed=42) if task == 'classification' else H2OReg(max_models=3, seed=42)
                models_to_eval['h2o'] = h2o_model
            except Exception: pass

            for m_name, model in models_to_eval.items():
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
        script_content = create_template_script(dataset_path, target_col, best_model, test_path, custom_metric)
        with open("train_model.py", "w") as f:
            f.write(script_content)
            
        return best_score, "train_model.py", task

    except Exception as e:
        log_error("Baseline evaluation failed", e)
        raise
