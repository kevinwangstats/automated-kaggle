import pandas as pd
import json
from logger import log_stage, log_error
import os

def perform_eda(dataset_path: str, output_path: str = "EDA.md"):
    log_stage("Automated EDA")
    try:
        df = pd.read_csv(dataset_path)
        
        shape = df.shape
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        missing_cols = missing_pct[missing_pct > 0].to_dict()
        
        # Categorical cardinality
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cardinality = {col: df[col].nunique() for col in cat_cols}
        
        # Skewness for numericals
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        skewness = df[num_cols].skew().round(2).to_dict()
        highly_skewed = {k: v for k, v in skewness.items() if abs(v) > 1.0}

        # Determine potential target (assuming last column if not specified, but let's just log columns)
        # We don't know the exact target variable, so we provide an overview.
        cols_info = {
            "Total Rows": shape[0],
            "Total Columns": shape[1],
            "Columns List": list(df.columns)
        }

        eda_content = [
            "# Exploratory Data Analysis",
            "",
            "## Dataset Shape",
            f"- Rows: {shape[0]}",
            f"- Columns: {shape[1]}",
            "",
            "## Columns",
            f"```json\n{json.dumps(list(df.columns), indent=2)}\n```",
            "",
            "## Missing Values (>0%)",
            f"```json\n{json.dumps(missing_cols, indent=2)}\n```",
            "",
            "## Categorical Cardinality",
            f"```json\n{json.dumps(cardinality, indent=2)}\n```",
            "",
            "## Highly Skewed Features (|skew| > 1)",
            f"```json\n{json.dumps(highly_skewed, indent=2)}\n```"
        ]
        
        with open(output_path, "w") as f:
            f.write("\n".join(eda_content))
            
        return output_path
    
    except Exception as e:
        log_error("Failed to perform EDA", e)
        raise
