import argparse
import yaml
from eda_engine import perform_eda
from baseline_engine import evaluate_baselines
from agent_loop import run_agent_loop
from git_manager import GitManager
from logger import log_stage, log_error

def main():
    parser = argparse.ArgumentParser(description="Agentic AutoML Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip user confirmation before LLM calls")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        dataset_path = config.get('dataset_path')
        target_col = config.get('target_col')
        test_path = config.get('test_path')
        metric = config.get('metric')
        iterations = config.get('iterations', 5)
        timeout = config.get('timeout', 600)
        
        if not dataset_path or not target_col:
            raise ValueError("Configuration file must contain 'dataset_path' and 'target_col'")

        git_mgr = GitManager()

        # Phase 1: EDA
        eda_path = perform_eda(dataset_path)

        # Phase 2: Baseline
        base_score, script_path, task = evaluate_baselines(
            dataset_path=dataset_path, 
            target_col=target_col,
            test_path=test_path,
            custom_metric=metric
        )
        
        # Initial commit to secure baseline state
        git_mgr.commit_all(f"Initial Baseline Commit | CV Score: {base_score:.4f}")

        # Phase 3: Agentic Loop
        run_agent_loop(
            dataset_path=dataset_path,
            target_col=target_col,
            base_score=base_score,
            git_mgr=git_mgr,
            task=task,
            max_iterations=iterations,
            skip_confirmation=args.yes,
            timeout=timeout
        )
        
    except Exception as e:
        log_error("Pipeline failed with a critical error", e)

if __name__ == "__main__":
    main()
