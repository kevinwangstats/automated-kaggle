import os
import json
import re
import subprocess
import yaml
from litellm import completion
from logger import log_stage, log_error, log_metric
from git_manager import GitManager

def extract_python_code(text: str) -> str:
    # Try to extract from markdown python block
    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1)
    # If no markdown block, return the whole text assuming it's pure python
    return text

def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()

def run_training_script(script_path="train_model.py", timeout: int = 600, config_path="config.yaml"):
    # Ensure no stale metrics exist
    if os.path.exists("metrics.json"):
        os.remove("metrics.json")
        
    # Run the script as a subprocess
    try:
        result = subprocess.run(
            ["python", script_path, "--config", config_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Script Execution Timed Out after {timeout} seconds.")
    
    if result.returncode != 0:
        raise RuntimeError(f"Script Execution Failed:\n{result.stderr}")
        
    # Attempt to format submission if it exists, only if auto_kaggle_submit is true
    auto_submit = False
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                auto_submit = cfg.get("auto_kaggle_submit", False)
        except Exception:
            pass
            
    # Always attempt to format the submission if it was generated
    if os.path.exists("raw_submission.csv"):
        subprocess.run(["python", "kaggle_submit.py", "--config", config_path, "--format-only"], capture_output=True)
        
    # Parse final score from metrics.json
    if not os.path.exists("metrics.json"):
        raise ValueError(f"Could not find metrics.json after script execution. Output was:\n{result.stdout}\n{result.stderr}")
        
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        if "cv_score" not in metrics:
            raise ValueError(f"metrics.json missing 'cv_score' key: {metrics}")
        return float(metrics["cv_score"])
    except json.JSONDecodeError:
        raise ValueError(f"metrics.json is malformed. Content: {open('metrics.json', 'r').read()}")

def run_agent_loop(
    dataset_path: str,
    target_col: str,
    base_score: float,
    git_mgr: GitManager,
    task: str,
    dataset_branch: str,
    max_iterations: int = 5,
    skip_confirmation: bool = False,
    timeout: int = 600,
    model: str = None,
    ollama_base_url: str = None,
    wandb_enabled: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    pred_prob: bool = True,
    config_path: str = "config.yaml"
):
    log_stage("Starting Agentic Loop")
    current_best_score = base_score
    history = []
    
    # Load history if exists
    if os.path.exists("history.json"):
        try:
            with open("history.json", "r") as f:
                history = json.load(f)
                
            if base_score is None and history:
                higher_is_better = (task == 'classification')
                valid_scores = [run['score'] for run in history if run.get('score') is not None]
                if valid_scores:
                    current_best_score = max(valid_scores) if higher_is_better else min(valid_scores)
        except:
            pass

    if current_best_score is None:
        raise ValueError("base_score was None and could not be determined from history.")

    eda_content = read_file("EDA.md")
    
    start_iteration = len(history) + 1
    end_iteration = start_iteration + max_iterations
    
    for i in range(start_iteration, end_iteration):
        log_stage(f"Iteration {i}")
        
        # We ensure we are on the dataset branch, but we DO NOT revert changes.
        # This allows a broken script from a failed previous iteration to persist
        # so the LLM can read it and attempt to fix its own errors.
        git_mgr.checkout_branch(dataset_branch)
        
        current_script = read_file("train_model.py")
        history_context = ""
        if len(history) > 0:
            last_run = history[-1]
            if last_run.get('error'):
                history_context = f"\nPREVIOUS RUN FAILED WITH ERROR:\n{last_run['error']}\nPlease fix this error in the current script.\n"
            elif not last_run.get('improved'):
                # If it ran but didn't improve, we might want to keep the new logic but tweak it, 
                # or the prompt will just tell it to try another approach on the existing script.
                history_context = f"\nPREVIOUS RUN DEGRADED SCORE ({last_run.get('score')} vs Best: {current_best_score}). Try a different approach.\n"
            else:
                # If the previous run improved, the dataset branch was already merged and is clean,
                # so the current_script is the new baseline.
                history_context = f"\nPREVIOUS RUN IMPROVED SCORE TO {last_run.get('score')}. Good job, keep going!\n"

        pred_prob_instruction = "Ensure that for the final `raw_submission.csv`, you predict the continuous PROBABILITIES for the positive class (e.g., using `predict_proba(test_X)[:, 1]`). Another script will handle formatting it for Kaggle into `submission.csv`."

        prompt = f"""You are an expert AI Data Scientist. Your goal is to improve the Cross-Validation score of the model.

CRITICAL: Your script MUST remain dataset-agnostic. 
- ALWAYS read `dataset_path`, `target_col`, and `test_path` from the configuration file.
- Support a `--config` command-line argument (using `argparse`) to specify the configuration file path (defaulting to `config.yaml`).
- NEVER hardcode column names (like "Survived") or file paths (like "data/titanic/train.csv").
- Use the `target_col` variable from the config for all target-related operations, including the submission file column name.
- When reading the dataset, you MUST preserve the `nrows=...` argument in `pd.read_csv` to prevent Out-Of-Memory crashes during evaluation.

MODELING FREEDOM: You are NOT restricted to the current model setup (e.g., CatBoost). 
- You are encouraged to change the model architecture, introduce ensembling (using `VotingClassifier`/`Regressor` or `StackingClassifier`/`Regressor`), or try different frameworks (XGBoost, LightGBM, CatBoost, H2O AutoML) to improve the score.
- You can add feature engineering, handle missing values better, and tune hyperparameters.

You have access to the following pre-configured models from the registry: {available_models}. You may tune their hyperparameters, but do not hallucinate imports for models outside of this list unless you are confident they are in the environment.

You should try tuning hyperparameters for these models, comparing their individual performance, or ensembling them (e.g., using `VotingClassifier`/`VotingRegressor` or `StackingClassifier`/`StackingRegressor`) to maximize the cross-validation score.

Dataset EDA Summary:
{eda_content}

Current Best Script:
```python
{current_script}
```

Current Best Score: {current_best_score}
{history_context}
Please propose a modified version of the Python script to improve the model. 
You can add feature engineering, handle missing values better, tune hyperparameters, or change the model architecture.
Always ensure you write the final cross-validation score to a file named `metrics.json` with the format: `{{"cv_score": final_score}}`.
{pred_prob_instruction}
Output ONLY the full modified Python code wrapped in ```python ... ``` blocks. Do not include other text.
"""
        
        if not skip_confirmation:
            print(f"\n[AgenticAutoML] Preparing to call LLM for iteration {i}.")
            confirm = input("Continue with this API call? (y/n): ")
            if confirm.lower() != 'y':
                print("Skipping LLM call and aborting loop.")
                break
                
        # Call LLM
        try:
            # Priority: config.yaml 'model' > AUTOML_MODEL env var > default
            model_name = model or os.environ.get("AUTOML_MODEL", "gemini/gemini-2.5-flash-lite")
            log_stage(f"Calling LLM: {model_name}")
            
            completion_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "request_timeout": 120  # Ensure LLM call doesn't hang indefinitely
            }
            # Ollama requires an api_base pointing to the local server
            if model_name.startswith("ollama"):
                base = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                completion_kwargs["api_base"] = base
                
                print(f"  [Info] Local Ollama model detected. Checking connection to {base} ...")
                import urllib.request
                import urllib.error
                try:
                    urllib.request.urlopen(base, timeout=3)
                except urllib.error.URLError:
                    print(f"  [Warning] Could not connect to Ollama at {base}.")
                    print(f"  [Warning] Please ensure the Ollama app is running locally.")
                print(f"  [Info] Waiting for Ollama response... (this may take a while depending on your hardware)")
            else:
                provider = model_name.split('/')[0] if '/' in model_name else model_name
                print(f"  [Info] Using remote API ({provider}). Please ensure your {provider.upper()}_API_KEY is set if you haven't.")
                print(f"  [Info] Waiting for API response...")
            
            response = completion(**completion_kwargs)
            llm_output = response.choices[0].message.content
        except Exception as e:
            log_error("LLM API Call failed", e)
            if "ollama" in (model_name or "").lower():
                print(f"  [Help] If you haven't pulled this model yet, open a new terminal and run: ollama pull {model_name.replace('ollama/', '')}")
            continue
            
        new_code = extract_python_code(llm_output)
        
        # Extract summary/reasoning for W&B
        llm_summary = re.sub(r'```python.*?```', '', llm_output, flags=re.DOTALL).strip()
        if not llm_summary:
            llm_summary = "No reasoning provided by LLM."
        
        # Write new code directly on the dataset branch
        with open("train_model.py", "w") as f:
            f.write(new_code)
            
        try:
            log_stage(f"Evaluating Generated Code")
            new_score = run_training_script("train_model.py", timeout=timeout, config_path=config_path)
            log_metric("Iteration Score", new_score)
            
            higher_is_better = (task == 'classification')
            
            improved = (new_score > current_best_score) if higher_is_better else (new_score < current_best_score)
            
            if wandb_enabled:
                import wandb
                
                wandb.log({
                    "cv_score": new_score, 
                    "improved": improved,
                    "iteration": len(history) + 1,
                    "model_used": model_name,
                    "llm_summary": wandb.Html(f"<pre>{llm_summary}</pre>"),
                    "prompt": wandb.Html(f"<pre>{prompt}</pre>")
                })
            
            if improved:
                log_stage(f"Score improved! ({current_best_score:.4f} -> {new_score:.4f})")
                current_best_score = new_score
                
                # Commit directly to the dataset branch
                commit_id = git_mgr.commit_all(f"[Iter {len(history)+1} | CV Score: {new_score:.4f}] Successful agent iteration")
                
                # Append Changelog
                with open("CHANGELOG.md", "a") as f:
                    f.write(f"\n- **Iter {len(history)+1}**: Score {new_score:.4f} (Commit: {commit_id})\n")
                    
                # Update history
                history.append({
                    "iteration": len(history)+1,
                    "commit": commit_id,
                    "score": new_score,
                    "improved": True,
                    "prompt": prompt,
                    "response": llm_output
                })
            else:
                log_stage(f"Score degraded or unchanged. Leaving changes uncommitted in workspace for next iteration to retry.")
                history.append({
                    "iteration": len(history)+1,
                    "commit": None,
                    "score": new_score,
                    "improved": False,
                    "prompt": prompt,
                    "response": llm_output
                })
                
        except Exception as e:
            log_error(f"Execution failed for iteration {i}", e)
            history.append({
                "iteration": len(history)+1,
                "commit": None,
                "score": None,
                "improved": False,
                "error": str(e),
                "prompt": prompt,
                "response": llm_output
            })
            
        with open("history.json", "w") as f:
            json.dump(history, f, indent=2)

    log_stage("Agentic Loop Finished")
    log_metric("Final Best Score", current_best_score)

