import os
import yaml
from pathlib import Path
import pandas as pd
import re

def load_config(config_path="config.yaml"):
    """Loads a YAML configuration and resolves dataset/test paths against REPO_ROOT."""
    config_p = Path(config_path)
    if not config_p.exists():
        return {}
        
    with open(config_p, "r") as f:
        config = yaml.safe_load(f) or {}
        
    # Resolve relative dataset paths against the repo root (passed via env var),
    # NOT the config file directory, since configs may live in subdirectories.
    repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd()))
    
    if config.get("dataset_path") and not Path(config.get("dataset_path")).is_absolute():
        config["dataset_path"] = str(repo_root / config["dataset_path"])
    if config.get("test_path") and not Path(config.get("test_path")).is_absolute():
        config["test_path"] = str(repo_root / config["test_path"])
        
    return config

def read_file(filepath: str) -> str:
    """Reads and returns the contents of a file as a string."""
    with open(filepath, 'r') as f:
        return f.read()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names to avoid errors in frameworks like LightGBM."""
    df.columns = [re.sub(r'[^\w\s]', '', str(col)).replace(' ', '_') for col in df.columns]
    return df
