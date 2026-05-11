import argparse
from eda_engine import perform_eda
from baseline_engine import evaluate_baselines
from agent_loop import run_agent_loop
from git_manager import GitManager
from logger import log_stage, log_error

def main():
    parser = argparse.ArgumentParser(description="Agentic AutoML Pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Kaggle CSV dataset")
    parser.add_argument("--target_col", type=str, required=True, help="Target column for prediction")
    parser.add_argument("--test_path", type=str, default=None, help="Optional path to test.csv for submission generation")
    parser.add_argument("--metric", type=str, default=None, help="Optional custom sklearn metric name (e.g. log_loss, f1)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for agent-generated script execution")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip user confirmation before LLM calls")
    parser.add_argument("--iterations", type=int, default=5, help="Number of agent iterations")
    args = parser.parse_args()

    try:
        git_mgr = GitManager()

        # Phase 1: EDA
        eda_path = perform_eda(args.dataset_path)

        # Phase 2: Baseline
        base_score, script_path, task = evaluate_baselines(
            dataset_path=args.dataset_path, 
            target_col=args.target_col,
            test_path=args.test_path,
            custom_metric=args.metric
        )
        
        # Initial commit to secure baseline state
        git_mgr.commit_all(f"Initial Baseline Commit | CV Score: {base_score:.4f}")

        # Phase 3: Agentic Loop
        run_agent_loop(
            dataset_path=args.dataset_path,
            target_col=args.target_col,
            base_score=base_score,
            git_mgr=git_mgr,
            task=task,
            max_iterations=args.iterations,
            skip_confirmation=args.yes,
            timeout=args.timeout
        )
        
    except Exception as e:
        log_error("Pipeline failed with a critical error", e)

if __name__ == "__main__":
    main()
