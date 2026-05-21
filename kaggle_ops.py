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

def setup_workspace(dataset_name: str):
    """
    Automates the Git worktree setup for a new dataset experiment.
    Installs requirements, creates the dataset branch, and builds the sibling worktree directory.
    """
    print(f"[Setup] Starting setup for dataset: {dataset_name}")
    
    # Install requirements
    print("[Setup] Installing requirements...")
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Setup] Error installing requirements: {e}")
        sys.exit(1)

    # Create branch
    print(f"[Setup] Creating branch '{dataset_name}'...")
    try:
        subprocess.run(["git", "branch", dataset_name], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"[Setup] Branch '{dataset_name}' likely already exists. Continuing.")

    # Create worktree
    project_root = Path(__file__).resolve().parent
    worktree_path = project_root.parent / f"automated-kaggle-{dataset_name}"
    
    print(f"[Setup] Creating worktree at '{worktree_path}'...")
    try:
        subprocess.run(["git", "worktree", "add", str(worktree_path), dataset_name], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"[Setup] Worktree at '{worktree_path}' likely already exists. Continuing.")

    # Download and extract dataset
    print(f"[Setup] Downloading Kaggle dataset '{dataset_name}'...")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", dataset_name, "-p", f"data/{dataset_name}", "--force"],
            cwd=str(worktree_path),
            check=True
        )
        print(f"[Setup] Extracting dataset '{dataset_name}'...")
        subprocess.run(
            ["unzip", "-o", f"data/{dataset_name}/{dataset_name}.zip", "-d", f"data/{dataset_name}/"],
            cwd=str(worktree_path),
            check=True,
            capture_output=True
        )
        print(f"[Setup] Cleaning up zip file...")
        subprocess.run(
            ["rm", f"data/{dataset_name}/{dataset_name}.zip"],
            cwd=str(worktree_path),
            check=True
        )
        print("[Setup] Data successfully downloaded and extracted.")
    except subprocess.CalledProcessError as e:
        print(f"[Setup] Warning: Data download/extraction failed. You may need to fetch it manually. Error: {e}")
    except FileNotFoundError:
        print("[Setup] Warning: 'kaggle' or 'unzip' command not found. Please ensure they are installed.")

    print(f"[Setup] Workspace ready! To begin, run: cd {worktree_path}")

def get_submission_path(config):
    """
    Determines where submission.csv should be saved.
    Default is "submission.csv" in the current directory.
    If example_submission is provided, it attempts to save it in the same directory,
    unless that would overwrite the example_submission itself.
    """
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
    config_p = Path(config_path)
    if not config_p.exists():
        print(f"[kaggle_ops] Config file {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_sub_path = Path(workspace_mgr.get_file_path("raw_submission.csv")) if workspace_mgr else Path("raw_submission.csv")
    if not raw_sub_path.exists():
        print(f"[kaggle_ops] {raw_sub_path} not found. Ensure the training script outputs this file.")
        return

    raw_sub = pd.read_csv(raw_sub_path)
    
    example_sub_path = config.get("example_submission", None)
    # The pred_prob setting from config is our source of truth.
    # raw_submission.csv is expected to always contain probabilities for classification.
    pred_prob = config.get("pred_prob", True)

    if example_sub_path and Path(example_sub_path).exists():
        print(f"[kaggle_ops] Reading example submission from {example_sub_path} for column names.")
    else:
        print(f"[kaggle_ops] Using config pred_prob: {pred_prob}")

    final_sub = raw_sub.copy()
    
    # If we have an example submission, ensure the column names and types match
    if example_sub_path and Path(example_sub_path).exists():
        example_sub = pd.read_csv(example_sub_path, nrows=1)
        if len(example_sub.columns) > 1 and len(final_sub.columns) > 1:
            # Match ID column name
            id_col_name = example_sub.columns[0]
            current_id_name = final_sub.columns[0]
            if current_id_name != id_col_name:
                print(f"[kaggle_ops] Renaming ID column from '{current_id_name}' to '{id_col_name}'")
                final_sub = final_sub.rename(columns={current_id_name: id_col_name})
                
            # Match target column name
            target_col_name = example_sub.columns[1]
            current_target_name = final_sub.columns[1]
            if current_target_name != target_col_name:
                print(f"[kaggle_ops] Renaming target column from '{current_target_name}' to '{target_col_name}'")
                final_sub = final_sub.rename(columns={current_target_name: target_col_name})
            
            target_col = target_col_name

    if len(final_sub.columns) > 1:
        target_col = final_sub.columns[1]
        if not pred_prob:
            print("[kaggle_ops] Converting probabilities to discrete classes (threshold=0.5).")
            # If probabilities are between 0 and 1, convert to 0/1
            if pd.api.types.is_float_dtype(final_sub[target_col]) and final_sub[target_col].between(0, 1).all():
                final_sub[target_col] = (final_sub[target_col] >= 0.5).astype(int)

    output_path = workspace_mgr.get_file_path("submission.csv") if workspace_mgr else get_submission_path(config)
    final_sub.to_csv(output_path, index=False)
    print(f"[kaggle_ops] Saved final formatted submission to {output_path}")

def submit_to_kaggle(config_path="config.yaml", commit_id=None, workspace_mgr=None):
    """
    Submits the formatted submission to Kaggle if auto_kaggle_submit is enabled.
    Includes the git commit ID in the submission message for provenance.
    """
    config_p = Path(config_path)
    if not config_p.exists():
        print(f"[kaggle_ops] Config file {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    auto_submit_val = str(config.get("auto_kaggle_submit", "never")).lower()
    if auto_submit_val in ["false", "never"]:
        print("[kaggle_ops] auto_kaggle_submit is false or never. Skipping automated Kaggle submission.")
        return

    sub_path = Path(workspace_mgr.get_file_path("submission.csv")) if workspace_mgr else Path(get_submission_path(config))
    if not sub_path.exists():
        print(f"[kaggle_ops] {sub_path} not found. Cannot submit.")
        return

    dataset_path = config.get("dataset_path", "")
    # Try to infer competition from dataset_path, e.g., data/titanic/train.csv -> titanic
    path_parts = dataset_path.split("/")
    competition_name = None
    if len(path_parts) >= 3 and path_parts[0] == "data":
        competition_name = path_parts[1]
    
    if not competition_name:
        print(f"[kaggle_ops] Could not infer competition name from dataset_path: {dataset_path}")
        return

    if not commit_id:
        try:
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()[:7]
        except Exception:
            commit_id = "unknown"
    else:
        commit_id = str(commit_id)[:7]

    submission_message = f"Agentic AutoML Pipeline Submission (Commit: {commit_id})"

    print(f"[kaggle_ops] Submitting to Kaggle competition: {competition_name} with message: '{submission_message}'")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", competition_name, "-f", sub_path, "-m", submission_message],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"[kaggle_ops] Errors/Warnings:\n{result.stderr}")
    except FileNotFoundError:
        print("[kaggle_ops] 'kaggle' command not found. Ensure kaggle CLI is installed and configured.")
    except Exception as e:
        print(f"[kaggle_ops] Failed to submit to Kaggle: {e}")

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
