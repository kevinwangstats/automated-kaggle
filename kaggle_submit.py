import pandas as pd
import yaml
import os
import argparse

def format_submission(config_path="config.yaml"):
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
    pred_prob = config.get("pred_prob", True)

    if example_sub_path and os.path.exists(example_sub_path):
        print(f"[kaggle_submit] Reading example submission from {example_sub_path}")
        example_sub = pd.read_csv(example_sub_path, nrows=10)
        if len(example_sub.columns) > 1:
            target_col = example_sub.columns[1]
            dtype = example_sub[target_col].dtype
            
            if pd.api.types.is_float_dtype(dtype):
                print("[kaggle_submit] Example submission uses floats. Outputting probabilities.")
                pred_prob = True
            elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                print("[kaggle_submit] Example submission uses integers/strings. Outputting discrete classes.")
                pred_prob = False
    else:
        print(f"[kaggle_submit] No example submission found. Using config pred_prob: {pred_prob}")

    final_sub = raw_sub.copy()
    
    if len(final_sub.columns) > 1:
        target_col = final_sub.columns[1]
        if not pred_prob:
            print("[kaggle_submit] Converting probabilities to discrete classes (threshold=0.5).")
            # If probabilities are between 0 and 1, convert to 0/1
            if pd.api.types.is_float_dtype(final_sub[target_col]) and final_sub[target_col].between(0, 1).all():
                final_sub[target_col] = (final_sub[target_col] >= 0.5).astype(int)

    final_sub.to_csv("submission.csv", index=False)
    print("[kaggle_submit] Saved final formatted submission to submission.csv")

def submit_to_kaggle(config_path="config.yaml"):
    import subprocess
    if not os.path.exists(config_path):
        print(f"[kaggle_submit] Config file {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config.get("auto_kaggle_submit", False):
        print("[kaggle_submit] auto_kaggle_submit is false or not set. Skipping automated Kaggle submission.")
        return

    if not os.path.exists("submission.csv"):
        print("[kaggle_submit] submission.csv not found. Cannot submit.")
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

    print(f"[kaggle_submit] Submitting to Kaggle competition: {competition_name}")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", competition_name, "-f", "submission.csv", "-m", "Agentic AutoML Pipeline Submission"],
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
    args = parser.parse_args()
    format_submission(args.config)
