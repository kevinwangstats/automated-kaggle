import optuna
import joblib
import json
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from logger import log_stage, log_info, log_error
import pandas as pd
import numpy as np

def run_optuna_optimization(workspace_mgr, trials=50, metric=None):
    """
    Loads preprocessed data and model type, runs Optuna hyperparameter tuning,
    and saves the best parameters to best_params.json.
    """
    log_stage(f"Optuna Hyperparameter Optimization ({trials} trials)")
    
    data_path = Path(workspace_mgr.get_file_path("optuna_data.pkl")) if workspace_mgr else Path("optuna_data.pkl")
    model_info_path = Path(workspace_mgr.get_file_path("optuna_model.txt")) if workspace_mgr else Path("optuna_model.txt")
    out_params_path = Path(workspace_mgr.get_file_path("best_params.json")) if workspace_mgr else Path("best_params.json")
    
    if not data_path.exists() or not model_info_path.exists():
        log_info("Optuna handoff artifacts (optuna_data.pkl, optuna_model.txt) not found. Skipping optimization.")
        return
        
    try:
        X_prep, y = joblib.load(data_path)
        with open(model_info_path, "r") as f:
            model_class = f.read().strip()
    except Exception as e:
        log_error("Failed to load Optuna handoff artifacts", e)
        return
        
    log_info(f"Loaded preprocessed data of shape {X_prep.shape} for model: {model_class}")
    
    # Determine task based on unique values
    # We use numpy for unique check because y might be numpy array or pandas series
    y_arr = np.array(y)
    unique_vals = len(np.unique(y_arr))
    task = 'classification' if unique_vals < 20 else 'regression'
    
    # Setup CV and default metric
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = metric if metric else 'roc_auc'
        direction = 'maximize'
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = metric if metric else 'neg_mean_squared_error'
        direction = 'maximize' if 'neg_' in scoring else 'minimize'
        # Actually scikit-learn cross_val_score returns negative metrics for losses, so we typically maximize them
        direction = 'maximize'

    def objective(trial):
        if model_class == 'LGBMClassifier':
            from lightgbm import LGBMClassifier
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }
            model = LGBMClassifier(**params)
            
        elif model_class == 'XGBClassifier':
            from xgboost import XGBClassifier
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'n_jobs': -1,
                'random_state': 42
            }
            model = XGBClassifier(**params)
            
        elif model_class == 'CatBoostClassifier':
            from catboost import CatBoostClassifier
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'verbose': 0,
                'thread_count': -1,
                'random_state': 42
            }
            model = CatBoostClassifier(**params)
            
        else:
            # Fallback for unsupported models
            raise optuna.exceptions.TrialPruned(f"Optuna tuning not implemented for {model_class}")

        scores = cross_val_score(model, X_prep, y, cv=cv, scoring=scoring, n_jobs=1)
        return scores.mean()

    # Create Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    
    try:
        study.optimize(objective, n_trials=trials, timeout=600)
    except Exception as e:
        log_info(f"Optimization interrupted or failed: {e}")
        
    best_score = study.best_value
    best_params = study.best_params
    
    log_info(f"Optimization completed. Best {scoring}: {best_score:.4f}")
    
    with open(out_params_path, "w") as f:
        json.dump({
            "model": model_class,
            "best_score": best_score,
            "scoring": scoring,
            "best_params": best_params
        }, f, indent=2)
        
    log_info(f"Saved best parameters to {out_params_path}")
    return best_score, best_params

if __name__ == "__main__":
    run_optuna_optimization(None, trials=5)
