import argparse
import yaml
from eda_engine import perform_eda
from baseline_engine import evaluate_baselines
from agent_loop import run_agent_loop, run_training_script
from git_manager import GitManager, dataset_branch_from_dataset_path
from logger import log_stage, log_error

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
        ollama_base_url = config.get('ollama_base_url', None)
        
        wandb_config = config.get('wandb', {})
        wandb_enabled = wandb_config.get('enabled', False)
        wandb_entity = wandb_config.get('entity', 'kevinwangstats')
        
        if not dataset_path or not target_col:
            raise ValueError("Configuration file must contain 'dataset_path' and 'target_col'")

        import sys
        import os
        import pandas as pd
        git_mgr = GitManager()
        dataset_branch = dataset_branch_from_dataset_path(dataset_path)
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

        if had_commits:
            git_mgr.ensure_dataset_branch(dataset_branch)
            if not args.resume:
                git_mgr.revert_changes()

        wandb_project = dataset_branch

        if wandb_enabled:
            import wandb
            run_id_file = "wandb_run_id.txt"
            
            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": f"{dataset_branch}_run",
                "config": {
                    "dataset_path": dataset_path,
                    "target_col": target_col,
                    "metric": metric,
                    "iterations": iterations,
                    "model": model
                }
            }
            
            if args.resume and os.path.exists(run_id_file):
                with open(run_id_file, "r") as f:
                    saved_run_id = f.read().strip()
                if saved_run_id:
                    init_kwargs["id"] = saved_run_id
                    init_kwargs["resume"] = "must"
                    
            wandb.init(**init_kwargs)
            
            if not args.resume or not os.path.exists(run_id_file):
                with open(run_id_file, "w") as f:
                    f.write(wandb.run.id)

        if args.resume:
            print("Resuming from previous state. Skipping EDA and Baseline generation.")
            base_score = None
            
            # Determine task
            df = pd.read_csv(dataset_path)
            y = df[target_col].dropna()
            task = 'classification' if y.nunique() < 20 else 'regression'
        else:
            # Phase 1: EDA
            eda_path = perform_eda(dataset_path)

            # Phase 2: Baseline
            base_score, script_path, task = evaluate_baselines(
                dataset_path=dataset_path, 
                target_col=target_col,
                test_path=test_path,
                custom_metric=metric,
                wandb_enabled=wandb_enabled,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity
            )
            
            # Initial commit to secure baseline state
            git_mgr.commit_all(f"Initial Baseline Commit | CV Score: {base_score:.4f}")
            if not had_commits:
                git_mgr.ensure_dataset_branch_after_initial_commit(dataset_branch)

        # Phase 3: Agentic Loop
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
            ollama_base_url=ollama_base_url,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            pred_prob=pred_prob,
            config_path=args.config
        )
        
        # Ensure we have a raw_submission.csv if we have a test_path
        if test_path and not os.path.exists("raw_submission.csv"):
            log_stage("Generating Baseline Submission")
            try:
                run_training_script(timeout=timeout, config_path=args.config)
            except Exception as e:
                log_error("Failed to generate baseline submission", e)

        # Phase 4: Automated Kaggle Submission
        import kaggle_submit
        log_stage("Final Kaggle Submission")
        kaggle_submit.format_submission(args.config)
        kaggle_submit.submit_to_kaggle(args.config)
        
        if wandb_enabled:
            wandb.finish()
            
    except Exception as e:
        log_error("Pipeline failed with a critical error", e)
        if 'wandb_enabled' in locals() and wandb_enabled:
            import wandb
            wandb.finish()

if __name__ == "__main__":
    main()
