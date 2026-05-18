import pandas as pd
import yaml
import os
import argparse

def get_submission_path(config):
    """
    Determines where submission.csv should be saved.
    Default is "submission.csv" in the current directory.
    If example_submission is provided, it attempts to save it in the same directory,
    unless that would overwrite the example_submission itself.
    """
    example_sub_path = config.get("example_submission")
    output_path = "submission.csv"
    
    if example_sub_path:
        example_dir = os.path.dirname(example_sub_path)
        # If example_dir is empty (file is in root), use current dir
        if not example_dir:
            example_dir = "."
            
        potential_path = os.path.join(example_dir, "submission.csv")
        
        # Check if potential_path is the same as example_sub_path to avoid overwriting
        if os.path.abspath(potential_path) != os.path.abspath(example_sub_path):
            output_path = potential_path
            
    return output_path

def format_submission(config_path="config.yaml"):
    """
    Reads raw_submission.csv and formats it into submission.csv based on config.
    Uses example_submission to infer target column and data types if available.
    """
    if not os.path.exists(config_path):
        print(f"[kaggle_submit] Config file {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_sub_path = "raw_submission.csv"
    if not os.path.exists(raw_sub_path):
        print(f"[kaggle_submit] {raw_sub_path} not found. Ensure the training script outputs this file.")
        return

    raw_sub = pd.read_csv(raw_sub_path)
    
    example_sub_path = config.get("example_submission", None)
    # The pred_prob setting from config is our source of truth.
    # raw_submission.csv is expected to always contain probabilities for classification.
    pred_prob = config.get("pred_prob", True)

    if example_sub_path and os.path.exists(example_sub_path):
        print(f"[kaggle_submit] Reading example submission from {example_sub_path} for column names.")
    else:
        print(f"[kaggle_submit] Using config pred_prob: {pred_prob}")

    final_sub = raw_sub.copy()
    
    # If we have an example submission, ensure the column names and types match
    if example_sub_path and os.path.exists(example_sub_path):
        example_sub = pd.read_csv(example_sub_path, nrows=1)
        if len(example_sub.columns) > 1 and len(final_sub.columns) > 1:
            # Match ID column name
            id_col_name = example_sub.columns[0]
            current_id_name = final_sub.columns[0]
            if current_id_name != id_col_name:
                print(f"[kaggle_submit] Renaming ID column from '{current_id_name}' to '{id_col_name}'")
                final_sub = final_sub.rename(columns={current_id_name: id_col_name})
                
            # Match target column name
            target_col_name = example_sub.columns[1]
            current_target_name = final_sub.columns[1]
            if current_target_name != target_col_name:
                print(f"[kaggle_submit] Renaming target column from '{current_target_name}' to '{target_col_name}'")
                final_sub = final_sub.rename(columns={current_target_name: target_col_name})
            
            target_col = target_col_name

    if len(final_sub.columns) > 1:
        target_col = final_sub.columns[1]
        if not pred_prob:
            print("[kaggle_submit] Converting probabilities to discrete classes (threshold=0.5).")
            # If probabilities are between 0 and 1, convert to 0/1
            if pd.api.types.is_float_dtype(final_sub[target_col]) and final_sub[target_col].between(0, 1).all():
                final_sub[target_col] = (final_sub[target_col] >= 0.5).astype(int)

    output_path = get_submission_path(config)
    final_sub.to_csv(output_path, index=False)
    print(f"[kaggle_submit] Saved final formatted submission to {output_path}")

def submit_to_kaggle(config_path="config.yaml", commit_id=None):
    """
    Submits the formatted submission to Kaggle if auto_kaggle_submit is enabled.
    Includes the git commit ID in the submission message for provenance.
    """
    import subprocess
    if not os.path.exists(config_path):
        print(f"[kaggle_submit] Config file {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config.get("auto_kaggle_submit", False):
        print("[kaggle_submit] auto_kaggle_submit is false or not set. Skipping automated Kaggle submission.")
        return

    sub_path = get_submission_path(config)
    if not os.path.exists(sub_path):
        print(f"[kaggle_submit] {sub_path} not found. Cannot submit.")
        return

    dataset_path = config.get("dataset_path", "")
    # Try to infer competition from dataset_path, e.g., data/titanic/train.csv -> titanic
    path_parts = dataset_path.split("/")
    competition_name = None
    if len(path_parts) >= 3 and path_parts[0] == "data":
        competition_name = path_parts[1]
    
    if not competition_name:
        print(f"[kaggle_submit] Could not infer competition name from dataset_path: {dataset_path}")
        return

    if not commit_id:
        try:
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()[:7]
        except Exception:
            commit_id = "unknown"
    else:
        commit_id = str(commit_id)[:7]

    submission_message = f"Agentic AutoML Pipeline Submission (Commit: {commit_id})"

    print(f"[kaggle_submit] Submitting to Kaggle competition: {competition_name} with message: '{submission_message}'")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", competition_name, "-f", sub_path, "-m", submission_message],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"[kaggle_submit] Errors/Warnings:\n{result.stderr}")
    except FileNotFoundError:
        print("[kaggle_submit] 'kaggle' command not found. Ensure kaggle CLI is installed and configured.")
    except Exception as e:
        print(f"[kaggle_submit] Failed to submit to Kaggle: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--format-only", action="store_true", help="Only format the submission, do not submit.")
    parser.add_argument("--submit-only", action="store_true", help="Only submit the existing submission, do not re-format.")
    args = parser.parse_args()
    
    if args.submit_only:
        submit_to_kaggle(args.config)
    elif args.format_only:
        format_submission(args.config)
    else:
        # Default behavior: do both
        format_submission(args.config)
        submit_to_kaggle(args.config)
