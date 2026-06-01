"""
kaggle_ops.py

Formats and automatically submits predictions to Kaggle.
Can be run as an independent module to manually trigger a submission:
    python kaggle_ops.py --config config.yaml [--submit-only | --format-only]
"""
import pandas as pd
import yaml
import os
import subprocess
import argparse
import sys
from pathlib import Path
from logger import log_info
from utils import load_config
from sklearn.preprocessing import LabelEncoder

def setup_workspace(dataset_name: str):
    """
    Automates the Git worktree setup for a new dataset experiment.
    Installs requirements, creates the dataset branch, and builds the sibling worktree directory.
    """
    log_info(f"Starting setup for dataset: {dataset_name}")
    
    # Install requirements
    log_info("Installing requirements...")
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError as e:
        log_info(f"Error installing requirements: {e}")
        sys.exit(1)

    # Create branch
    log_info(f"Creating branch '{dataset_name}'...")
    try:
        subprocess.run(["git", "branch", dataset_name], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        log_info(f"Branch '{dataset_name}' likely already exists. Continuing.")

    # Create worktree
    project_root = Path(__file__).resolve().parent
    worktree_path = project_root.parent / f"automated-kaggle-{dataset_name}"
    
    log_info(f"Creating worktree at '{worktree_path}'...")
    try:
        subprocess.run(["git", "worktree", "add", str(worktree_path), dataset_name], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        log_info(f"Worktree at '{worktree_path}' likely already exists. Continuing.")

    # Download and extract dataset
    log_info(f"Downloading Kaggle dataset '{dataset_name}'...")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", dataset_name, "-p", f"data/{dataset_name}", "--force"],
            cwd=str(worktree_path),
            check=True
        )
        log_info(f"Extracting dataset '{dataset_name}'...")
        subprocess.run(
            ["unzip", "-o", f"data/{dataset_name}/{dataset_name}.zip", "-d", f"data/{dataset_name}/"],
            cwd=str(worktree_path),
            check=True,
            capture_output=True
        )
        log_info(f"Cleaning up zip file...")
        subprocess.run(
            ["rm", f"data/{dataset_name}/{dataset_name}.zip"],
            cwd=str(worktree_path),
            check=True
        )
        log_info("Data successfully downloaded and extracted.")
    except subprocess.CalledProcessError as e:
        log_info(f"Warning: Data download/extraction failed. You may need to fetch it manually. Error: {e}")
    except FileNotFoundError:
        log_info("Warning: 'kaggle' or 'unzip' command not found. Please ensure they are installed.")

    log_info(f"Workspace ready! To begin, run: cd {worktree_path}")

def get_submission_path(config, workspace_mgr=None):
    """
    Determines where submission.csv should be saved.
    Returns the workspace manager's path if available.
    Otherwise defaults to "submission.csv" or follows example_submission context.
    """
    if workspace_mgr:
        return str(Path(workspace_mgr.get_file_path("submission.csv")))
        
    example_sub_path = config.get("example_submission")
    output_path = Path("submission.csv")
    
    if example_sub_path:
        example_path = Path(example_sub_path)
        example_dir = example_path.parent
            
        potential_path = example_dir / "submission.csv"
        
        # Check if potential_path is the same as example_sub_path to avoid overwriting
        if potential_path.resolve() != example_path.resolve():
            output_path = potential_path
            
    return str(output_path)

def format_submission(config_path="config.yaml", workspace_mgr=None):
    """
    Reads raw_submission.csv and formats it into submission.csv based on config.
    Uses example_submission to infer target column and data types if available.
    """
    config = load_config(config_path)
    if not config:
        log_info(f"Config file {config_path} not found or invalid.")
        return

    raw_sub_path = Path(workspace_mgr.get_file_path("raw_submission.csv")) if workspace_mgr else Path("raw_submission.csv")
    if not raw_sub_path.exists():
        log_info(f"{raw_sub_path} not found. Ensure the training script outputs this file.")
        return

    raw_sub = pd.read_csv(raw_sub_path)
    
    example_sub_path = config.get("example_submission", None)
    # The pred_type setting from config is our source of truth.
    # raw_submission.csv is expected to always contain probabilities for classification.
    pred_type = config.get("pred_type", "prob")

    if example_sub_path and Path(example_sub_path).exists():
        log_info(f"Reading example submission from {example_sub_path} for column names.")
    else:
        log_info(f"Using config pred_type: {pred_type}")

    final_sub = raw_sub.copy()
    
    # If we have an example submission, ensure the column names and types match
    if example_sub_path and Path(example_sub_path).exists():
        example_sub = pd.read_csv(example_sub_path, nrows=1)
        if len(example_sub.columns) > 1 and len(final_sub.columns) > 1:
            # Match ID column name
            id_col_name = example_sub.columns[0]
            current_id_name = final_sub.columns[0]
            if current_id_name != id_col_name:
                log_info(f"Renaming ID column from '{current_id_name}' to '{id_col_name}'")
                final_sub = final_sub.rename(columns={current_id_name: id_col_name})
                
            # Match target column name
            target_col_name = example_sub.columns[1]
            current_target_name = final_sub.columns[1]
            if current_target_name != target_col_name:
                log_info(f"Renaming target column from '{current_target_name}' to '{target_col_name}'")
                final_sub = final_sub.rename(columns={current_target_name: target_col_name})
            
            target_col = target_col_name

    if pred_type == "multiclass_class":
        log_info("Converting probabilities to discrete string classes via argmax.")
        dataset_path = config.get("dataset_path")
        target_col_name = config.get("target_col")
        
        train_df = pd.read_csv(dataset_path, usecols=[target_col_name])
        train_df = train_df.dropna(subset=[target_col_name])
        
        le = LabelEncoder()
        le.fit(train_df[target_col_name])
        
        prob_cols = final_sub.columns[1:]
        max_idx = final_sub[prob_cols].values.argmax(axis=1)
        predicted_labels = le.inverse_transform(max_idx)
        
        new_sub = pd.DataFrame()
        new_sub[final_sub.columns[0]] = final_sub.iloc[:, 0]
        
        # Determine the correct target column name for submission
        if example_sub_path and Path(example_sub_path).exists():
            final_target_name = example_sub.columns[1] if len(example_sub.columns) > 1 else target_col_name
        else:
            final_target_name = target_col_name
            
        new_sub[final_target_name] = predicted_labels
        final_sub = new_sub
        
    elif pred_type == "multiclass_prob":
        log_info("Mapping probability columns to match example_submission.")
        if example_sub_path and Path(example_sub_path).exists():
            if len(example_sub.columns) == len(final_sub.columns):
                col_map = {final_sub.columns[i]: example_sub.columns[i] for i in range(len(final_sub.columns))}
                final_sub = final_sub.rename(columns=col_map)
            else:
                log_info("Warning: Column count mismatch between raw_submission and example_submission.")
    elif len(final_sub.columns) > 1:
        target_col = final_sub.columns[1]
        if pred_type == "0/1":
            log_info("Converting probabilities to discrete 0/1 classes (threshold=0.5).")
            if pd.api.types.is_float_dtype(final_sub[target_col]) and final_sub[target_col].between(0, 1).all():
                final_sub[target_col] = (final_sub[target_col] >= 0.5).astype(int)
        elif pred_type == "true/false":
            log_info("Converting probabilities to True/False labels (threshold=0.5).")
            if pd.api.types.is_float_dtype(final_sub[target_col]) and final_sub[target_col].between(0, 1).all():
                final_sub[target_col] = (final_sub[target_col] >= 0.5)
        else:
            # pred_type == "prob" — keep raw probabilities as-is
            pass

    output_path = get_submission_path(config, workspace_mgr)
    final_sub.to_csv(output_path, index=False)
    log_info(f"Saved final formatted submission to {output_path}")

def submit_to_kaggle(config_path="config.yaml", commit_id=None, workspace_mgr=None):
    """
    Submits the formatted submission to Kaggle if auto_kaggle_submit is enabled.
    Includes the git commit ID in the submission message for provenance.
    """
    config = load_config(config_path)
    if not config:
        log_info(f"Config file {config_path} not found or invalid.")
        return

    auto_submit_val = str(config.get("auto_kaggle_submit", "never")).lower()
    if auto_submit_val in ["false", "never"]:
        log_info("auto_kaggle_submit is false or never. Skipping automated Kaggle submission.")
        return

    sub_path = Path(get_submission_path(config, workspace_mgr))
    if not sub_path.exists():
        log_info(f"{sub_path} not found. Cannot submit.")
        return

    dataset_path = config.get("dataset_path", "")
    # Try to infer competition from dataset_path, e.g., .../data/titanic/train.csv -> titanic
    competition_name = None
    try:
        p = Path(dataset_path)
        data_idx = p.parts.index("data")
        competition_name = p.parts[data_idx + 1]
    except (ValueError, IndexError):
        pass

    if not competition_name:
        log_info(f"Could not infer competition name from dataset_path: {dataset_path}")
        return

    if not commit_id:
        try:
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()[:7]
        except Exception:
            commit_id = "unknown"
    else:
        commit_id = str(commit_id)[:7]

    submission_message = f"Agentic AutoML Pipeline Submission (Commit: {commit_id})"

    log_info(f"Submitting to Kaggle competition: {competition_name} with message: '{submission_message}'")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", competition_name, "-f", sub_path, "-m", submission_message],
            capture_output=True,
            text=True
        )
        log_info(result.stdout)
        if result.stderr:
            log_info(f"Errors/Warnings:\n{result.stderr}")
    except FileNotFoundError:
        log_info("'kaggle' command not found. Ensure kaggle CLI is installed and configured.")
    except Exception as e:
        log_info(f"Failed to submit to Kaggle: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--format-only", action="store_true", help="Only format the submission, do not submit.")
    parser.add_argument("--submit-only", action="store_true", help="Only submit the existing submission, do not re-format.")
    parser.add_argument("--setup", type=str, help="Automates the Git worktree setup for a new dataset.")
    args = parser.parse_args()
    
    if args.setup:
        setup_workspace(args.setup)
        sys.exit(0)
    
    if args.submit_only:
        submit_to_kaggle(args.config)
    elif args.format_only:
        format_submission(args.config)
    else:
        # Default behavior: do both
        format_submission(args.config)
        submit_to_kaggle(args.config)
