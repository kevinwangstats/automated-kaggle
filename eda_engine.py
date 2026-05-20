"""
eda_engine.py

Performs automated Exploratory Data Analysis and generates EDA.md.
It is primarily imported by the orchestrator, but can be run via Python shell:
    >>> from eda_engine import perform_eda
    >>> perform_eda("data/train.csv", "target_col")
"""
import pandas as pd
import json
from logger import log_stage, log_error
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def perform_eda(dataset_path: str, target_col: str, output_path: str = "EDA.md", max_rows: int = None, workspace_mgr=None):
    log_stage("Enhanced Automated EDA")
    try:
        df = pd.read_csv(dataset_path, nrows=max_rows)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        shape = df.shape
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        missing_cols = missing_pct[missing_pct > 0].to_dict()
        
        # 1. Target Variable Profiling
        target_series = df[target_col].dropna()
        unique_targets = target_series.nunique()
        task = 'classification' if unique_targets < 20 else 'regression'
        
        target_info = {"Task": task}
        if task == 'classification':
            counts = target_series.value_counts()
            pcts = target_series.value_counts(normalize=True) * 100
            dist = {str(k): f"{counts[k]} ({pcts[k]:.2f}%)" for k in counts.index}
            target_info["Class Distribution"] = dist
        else:
            target_info["Stats"] = {
                "Min": float(target_series.min()),
                "Max": float(target_series.max()),
                "Mean": float(target_series.mean().round(4)),
                "Skew": float(target_series.skew().round(4))
            }

        # 2. Feature-to-Target Correlation (Numerical)
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        correlations = {}
        
        corr_df = df[num_cols].copy()
        # Ensure target is in corr_df for correlation computation
        if target_col not in corr_df.columns:
            if df[target_col].dtype == object or df[target_col].dtype.name == 'category' or df[target_col].dtype == bool:
                mode_vals = df[target_col].mode()
                fill_val = mode_vals[0] if not mode_vals.empty else "Missing"
                corr_df[target_col] = LabelEncoder().fit_transform(df[target_col].fillna(fill_val).astype(str))
            else:
                # Fallback for any other type, just try to coerce to numeric or drop
                try:
                    corr_df[target_col] = pd.to_numeric(df[target_col])
                except Exception:
                    pass

        if target_col in corr_df.columns and len(corr_df.columns) > 1:
            all_corrs = corr_df.corr()[target_col].drop(labels=[target_col]).dropna().sort_values(ascending=False)
            
            top_pos = {str(k): float(v) for k, v in all_corrs.head(5).round(4).items()}
            top_neg = {str(k): float(v) for k, v in all_corrs.tail(5).round(4).items()}
            correlations = {"Top 5 Positive": top_pos, "Top 5 Negative": top_neg}

        # 3. Data Signatures & Samples
        sample_df = df.sample(n=min(5, len(df)), random_state=42)
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_samples = {}
        for col in cat_cols:
            unique_vals = df[col].dropna().unique()
            cat_samples[col] = list(unique_vals[:5].astype(str))

        eda_content = [
            "# Exploratory Data Analysis",
            "",
            "## Dataset Overview",
            f"- Rows: {shape[0]}",
            f"- Columns: {shape[1]}",
            f"- Target Column: `{target_col}`",
            "",
            "## Target Profiling",
            f"```json\n{json.dumps(target_info, indent=2)}\n```",
            "",
            "## Feature-to-Target Correlation (Numerical)",
            f"```json\n{json.dumps(correlations, indent=2)}\n```",
            "",
            "## Missing Values (>0%)",
            f"```json\n{json.dumps(missing_cols, indent=2)}\n```",
            "",
            "## Categorical Value Samples (Top 5 Unique)",
            f"```json\n{json.dumps(cat_samples, indent=2)}\n```",
            "",
            "## Data Signature (Random 5-Row Sample)",
            "```csv",
            sample_df.to_csv(index=False),
            "```",
            ""
        ]

        
        with open(output_path, "w") as f:
            f.write("\n".join(eda_content))
            
        return output_path
    
    except Exception as e:
        log_error("Failed to perform enhanced EDA", e)
        raise
