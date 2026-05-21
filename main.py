"""
main.py

The main orchestrator for the Agentic AutoML pipeline.
Execute this script as the primary entry point:
    python main.py --config config.yaml [-y] [--resume]
"""
import argparse
import yaml
import sys
import os
import pandas as pd
import json
import re
import wandb
import kaggle_ops
from eda_engine import perform_eda
from baseline_engine import evaluate_baselines
from agent_loop import run_agent_loop, run_training_script
from git_manager import GitManager, dataset_branch_from_dataset_path
from workspace_manager import WorkspaceManager
from logger import log_stage, log_error
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Agentic AutoML Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip user confirmation before LLM calls")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume from previous iterations")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        dataset_path = config.get('dataset_path')
        target_col = config.get('target_col')
        test_path = config.get('test_path')
        metric = config.get('metric')
        pred_prob = config.get('pred_prob', True)
        iterations = config.get('iterations', 5)
        timeout = config.get('timeout', 600)
        model = config.get('model', None)
        temperature = config.get('temperature', 0.4)
        ollama_base_url = config.get('ollama_base_url', None)
        max_rows = config.get('max_rows', 100000)
        
        wandb_config = config.get('wandb', {})
        wandb_enabled = wandb_config.get('enabled', False)
        wandb_entity = wandb_config.get('entity', 'kevinwangstats')
        
        if not dataset_path or not target_col:
            raise ValueError("Configuration file must contain 'dataset_path' and 'target_col'")

        git_mgr = GitManager()
        dataset_branch = dataset_branch_from_dataset_path(dataset_path)
        
        workspace_root = config.get("workspace_root", ".workspaces")
        run_mode = config.get("run_mode", "prompt")
        
        workspace_mgr = WorkspaceManager(dataset_branch, root_dir=workspace_root)
        had_commits = bool(git_mgr.repo.heads)
        
        if had_commits and git_mgr.is_on_main() and dataset_branch != "main":
            if git_mgr.has_uncommitted_changes():
                print("\n[Warning] You have uncommitted changes on the 'main' branch.")
                print("You will not be running the latest software unless you commit.")
                ans = input("Are you happy to git add all file changes, commit, and push before proceeding to work on the dataset branch? (y/n): ")
                if ans.lower() == 'y':
                    msg = input("Enter commit message: ")
                    if not msg: msg = "Update core files"
                    git_mgr.commit_all(msg)
                    try:
                        git_mgr.repo.remotes.origin.push()
                        print("Pushed to origin.")
                    except Exception as e:
                        print(f"Push to origin skipped/failed: {e}")
                else:
                    print("Aborting. Please stash or commit your changes manually before proceeding to avoid conflicts.")
                    sys.exit(1)
            
            if git_mgr.branch_exists(dataset_branch) and not git_mgr.is_branch_based_on_latest_main(dataset_branch):
                print(f"\n[Warning] The dataset branch '{dataset_branch}' already exists, but it is not based off the latest commit on 'main'.")
                print("You will not be running the latest software for this dataset.")
                ans = input(f"Would you like to delete the '{dataset_branch}' branch by force to start fresh from the latest main? (y/n): ")
                if ans.lower() == 'y':
                    git_mgr.delete_branch(dataset_branch)
                    print(f"Deleted outdated dataset branch '{dataset_branch}'.")

        # Checkout dataset branch to reveal tracked files
        if had_commits:
            git_mgr.ensure_dataset_branch(dataset_branch)

        # State detection: check the ROOT directory for tracked artifacts
        has_previous_state = Path("history.json").exists() and Path("train_model.py").exists()
                
        should_resume = args.resume
        if has_previous_state and not args.resume:
            if args.yes:
                resolved_mode = "resume" if run_mode == "prompt" else run_mode
                should_resume = (resolved_mode == "resume")
                print(f"Skipping interactive prompt due to -y flag. Using fallback mode: {resolved_mode}")
            else:
                if run_mode == "resume":
                    should_resume = True
                elif run_mode == "scratch":
                    should_resume = False
                else:
                    ans = input("[Warning] Previous iterations detected for this dataset. Do you want to resume from the existing train_model.py and history? (y/n): ")
                    should_resume = (ans.lower() == 'y')
                
        if args.resume and not has_previous_state:
            print("[Warning] --resume passed but no previous root state found. Falling back to start from scratch.")
            should_resume = False

        if had_commits and not should_resume:
            git_mgr.revert_changes()
            # Clean up root files to start completely fresh
            for f in ["train_model.py", "history.json", "EDA.md"]:
                p = Path(f)
                if p.exists(): p.unlink()

        wandb_project = dataset_branch

        if wandb_enabled:
            run_id_file = Path(workspace_mgr.get_file_path("wandb_run_id.txt")) if workspace_mgr else Path("wandb_run_id.txt")
            
            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": f"{dataset_branch}_run",
                "config": {
                    "dataset_path": dataset_path,
                    "target_col": target_col,
                    "metric": metric,
                    "iterations": iterations,
                    "model": model,
                    "max_rows": max_rows
                }
            }
            
            if should_resume and run_id_file.exists():
                with open(run_id_file, "r") as f:
                    saved_run_id = f.read().strip()
                if saved_run_id:
                    init_kwargs["id"] = saved_run_id
                    init_kwargs["resume"] = "must"
                    
            wandb.init(**init_kwargs)
            
            if not should_resume or not run_id_file.exists():
                with open(run_id_file, "w") as f:
                    f.write(wandb.run.id)

        if should_resume:
            print("Resuming from previous state. Skipping EDA and Baseline generation.")
            base_score = None
            
            # Determine task
            df = pd.read_csv(dataset_path, nrows=max_rows)
            y = df[target_col].dropna()
            task = 'classification' if y.nunique() < 20 else 'regression'

            history_path = Path("history.json")
            script_path = Path("train_model.py")
            
            history = []
            if history_path.exists():
                try:
                    with open(history_path, "r") as f:
                        history = json.load(f)
                except Exception:
                    pass

            current_code = ""
            if script_path.exists():
                with open(script_path, "r") as f:
                    current_code = f.read().strip()
                    
            # 1. Detect if script was manually modified
            is_modified = False
            if history and current_code:
                last_entry = history[-1]
                last_response = last_entry.get("response", "")
                
                match = re.search(r'```python\n(.*?)\n```', last_response, re.DOTALL)
                last_code = match.group(1).strip() if match else last_response.strip()
                
                if current_code != last_code:
                    is_modified = True
                    print("[Info] Detected manual modifications to train_model.py.")
            
            # 2. Always execute train_model.py as-is first
            print("[Info] Executing existing train_model.py to establish current baseline score...")
            try:
                base_score = run_training_script(
                    script_path=script_path,
                    timeout=timeout,
                    config_path=args.config,
                    workspace_mgr=workspace_mgr
                )
                print(f"[Info] Established current score: {base_score:.4f}")
            except Exception as e:
                log_error("Failed to execute existing train_model.py", e)
                print("[Info] Falling back to regenerating a fresh baseline...")
                eda_path = perform_eda(dataset_path, target_col, max_rows=max_rows, workspace_mgr=workspace_mgr)
                base_score, script_path, task = evaluate_baselines(
                    dataset_path=dataset_path,
                    target_col=target_col,
                    test_path=test_path,
                    custom_metric=metric,
                    wandb_enabled=wandb_enabled,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    max_rows=max_rows,
                    workspace_mgr=workspace_mgr
                )
                git_mgr.commit_all(f"Baseline Regenerated (broken resume) | CV Score: {base_score:.4f}")
                print(f"[Info] Regenerated baseline score: {base_score:.4f}")
                # Skip human intervention logging since we regenerated
                is_modified = False

            # 3. Log HUMAN_INTERVENTION if modified, now using the evaluated score
            if is_modified:
                print("[Info] Logging HUMAN_INTERVENTION to history with the evaluated score.")
                valid_scores = [r['score'] for r in history if r.get('score') is not None]
                prev_best = max(valid_scores) if valid_scores else float('-inf')
                improved = (base_score > prev_best) if valid_scores else True

                synthetic_entry = {
                    "iteration": len(history) + 1,
                    "commit": git_mgr.get_current_commit() if had_commits else None,
                    "score": base_score,
                    "improved": improved,
                    "prompt": "HUMAN_INTERVENTION",
                    "response": f"User manually modified train_model.py prior to resuming.\n\n```python\n{current_code}\n```",
                    "error": None
                }
                history.append(synthetic_entry)
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)
                
                if improved and had_commits:
                    commit_msg = f"[Iter {len(history)} | CV Score: {base_score:.4f}] Manual Human Intervention"
                    git_mgr.commit_all(commit_msg)
                    try:
                        with open("CHANGELOG.md", "a") as f:
                            f.write(f"\n- **Iter {len(history)}**: Score {base_score:.4f} (Commit: {git_mgr.get_current_commit()}) [MANUAL]\n")
                    except Exception:
                        pass
            
            # 4. Auto-submit resumed state to Kaggle
            auto_submit_val = str(config.get("auto_kaggle_submit", "never")).lower()
            should_submit = False
            if auto_submit_val in ["always", "true"]:
                should_submit = True
            elif auto_submit_val == "best" and is_modified and improved:
                should_submit = True
                
            if should_submit:
                try:
                    raw_sub_path = Path(workspace_mgr.get_file_path("raw_submission.csv")) if workspace_mgr else Path("raw_submission.csv")
                    if raw_sub_path.exists():
                        log_stage(f"Automated Kaggle Submission for Resumed State")
                        kaggle_ops.format_submission(args.config, workspace_mgr=workspace_mgr)
                        kaggle_ops.submit_to_kaggle(args.config, commit_id=git_mgr.get_current_commit(), workspace_mgr=workspace_mgr)
                except Exception as e:
                    log_error(f"Failed to submit resumed state to Kaggle", e)
        else:
            # Phase 1: EDA
            eda_path = perform_eda(dataset_path, target_col, max_rows=max_rows, workspace_mgr=workspace_mgr)

            # Phase 2: Baseline
            base_score, script_path, task = evaluate_baselines(
                dataset_path=dataset_path, 
                target_col=target_col,
                test_path=test_path,
                custom_metric=metric,
                wandb_enabled=wandb_enabled,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                max_rows=max_rows,
                workspace_mgr=workspace_mgr
            )
            
            # Initial commit to secure baseline state
            git_mgr.commit_all(f"Initial Baseline Commit | CV Score: {base_score:.4f}")
            if not had_commits:
                git_mgr.ensure_dataset_branch_after_initial_commit(dataset_branch)

            # Auto-submit baseline to Kaggle
            auto_submit_val = str(config.get("auto_kaggle_submit", "never")).lower()
            if auto_submit_val in ["always", "true", "best"]:
                try:

                    raw_sub_path = Path(workspace_mgr.get_file_path("raw_submission.csv")) if workspace_mgr else Path("raw_submission.csv")
                    if raw_sub_path.exists():
                        log_stage(f"Automated Kaggle Submission for Baseline")
                        kaggle_ops.format_submission(args.config, workspace_mgr=workspace_mgr)
                        kaggle_ops.submit_to_kaggle(args.config, commit_id=git_mgr.get_current_commit(), workspace_mgr=workspace_mgr)
                except Exception as e:
                    log_error(f"Failed to submit baseline to Kaggle", e)

        # Phase 3: Agentic Loop
        available_models = []
        if Path("models_registry.yaml").exists():
            try:
                with open("models_registry.yaml", "r") as f:
                    reg = yaml.safe_load(f)
                    if reg and 'models' in reg:
                        available_models = list(reg['models'].keys())
            except Exception:
                pass

        run_agent_loop(
            dataset_path=dataset_path,
            target_col=target_col,
            base_score=base_score,
            git_mgr=git_mgr,
            task=task,
            dataset_branch=dataset_branch,
            max_iterations=iterations,
            skip_confirmation=args.yes,
            timeout=timeout,
            model=model,
            temperature=temperature,
            ollama_base_url=ollama_base_url,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            pred_prob=pred_prob,
            config_path=args.config,
            available_models=available_models,
            workspace_mgr=workspace_mgr
        )
        
        # Ensure we have a raw_submission.csv if we have a test_path
        raw_sub_path = Path(workspace_mgr.get_file_path("raw_submission.csv")) if workspace_mgr else Path("raw_submission.csv")
        if test_path and not raw_sub_path.exists():
            log_stage("Generating Baseline Submission")
            try:
                script_path = "train_model.py"
                run_training_script(script_path=script_path, timeout=timeout, config_path=args.config, workspace_mgr=workspace_mgr)
            except Exception as e:
                log_error("Failed to generate baseline submission", e)

        # Phase 4: Format Final Submission

        log_stage("Formatting Final Submission")
        kaggle_ops.format_submission(args.config, workspace_mgr=workspace_mgr)

        if wandb_enabled:
            wandb.finish()
            
    except Exception as e:
        log_error("Pipeline failed with a critical error", e)
        if 'wandb_enabled' in locals() and wandb_enabled:
            wandb.finish()

if __name__ == "__main__":
    main()
