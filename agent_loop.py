"""
agent_loop.py

This module orchestrates the LLM improvement cycle.
It is primarily designed to be imported and called by main.py.
However, it can be tested independently in a Python shell:
    >>> from agent_loop import run_agent_loop
    >>> run_agent_loop(...)
"""
import os
import json
import re
import subprocess
import yaml
import urllib.request
import urllib.error
import wandb
import weave
import kaggle_ops
from pathlib import Path
from litellm import completion
from litellm.exceptions import Timeout, BadRequestError, AuthenticationError
from logger import log_stage, log_error, log_metric
from git_manager import GitManager

def get_file_messages(file_paths: list, model_name: str, api_key: str = None) -> list:
    """Uploads files via native API if supported, or falls back to local text extraction."""
    from pathlib import Path
    messages = []
    
    # 1. Kimi / Moonshot / OpenAI
    if "kimi" in model_name.lower() or "moonshot" in model_name.lower() or "openai" in model_name.lower():
        if api_key:
            try:
                from openai import OpenAI
                base_url = "https://api.moonshot.ai/v1" if ("kimi" in model_name.lower() or "moonshot" in model_name.lower()) else None
                client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
                
                print(f"  [Info] Uploading files via OpenAI/Moonshot API...")
                for fp in file_paths:
                    if not Path(fp).exists(): continue
                    file_object = client.files.create(file=Path(fp), purpose="file-extract")
                    file_content = client.files.content(file_id=file_object.id).text
                    messages.append({"role": "system", "content": f"File Content for {fp}:\n{file_content}"})
                return messages
            except Exception as e:
                print(f"  [Warning] File upload failed: {e}. Falling back to local text reading.")
        else:
             print("  [Warning] API key not found. Falling back to local file reading.")

    # 2. Gemini
    elif "gemini" in model_name.lower():
        try:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            print(f"  [Info] Uploading files via Gemini API...")
            for fp in file_paths:
                if not Path(fp).exists(): continue
                file_object = genai.upload_file(fp)
                # LiteLLM expects the file object inside a list for the content
                messages.append({"role": "system", "content": [file_object, f"File: {fp}"]})
            return messages
        except Exception as e:
            print(f"  [Warning] Gemini file upload failed: {e}. Falling back to local text reading.")

    # 3. Fallback
    print("  [Info] Reading files locally as fallback...")
    for fp in file_paths:
        if Path(fp).exists():
            with open(fp, 'r') as f:
                messages.append({"role": "system", "content": f"File Content for {fp}:\n{f.read()}"})
                
    return messages

def extract_python_code(text: str) -> str:
    # Try to extract from markdown python block
    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # If no markdown block, return the whole text assuming it's pure python
        code = text.strip()
        
    if not code:
        raise ValueError("LLM generated an empty response or no valid Python code could be extracted.")
    return code

def read_file(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()

@weave.op()
def call_agent_llm(completion_kwargs: dict) -> str:
    """Wrapper for LLM completion to enable Weave tracing."""
    response = completion(**completion_kwargs)
    return response.choices[0].message.content

def run_training_script(script_path="train_model.py", timeout: int = 600, config_path="config.yaml", workspace_mgr=None):
    # Ensure no stale metrics exist
    metrics_path = Path(workspace_mgr.get_file_path("metrics.json")) if workspace_mgr else Path("metrics.json")
    if metrics_path.exists():
        metrics_path.unlink()
        
    # Run the script as a subprocess
    try:
        abs_config = Path(config_path).resolve() if config_path else Path("config.yaml").resolve()
        output_dir = Path(workspace_mgr.workspace_dir).resolve() if workspace_mgr else Path(".").resolve()
        
        # Pass REPO_ROOT so the generated script can resolve relative dataset paths
        # against the repository root, not the workspace or config file directory.
        env = os.environ.copy()
        env["REPO_ROOT"] = str(Path.cwd())
        
        result = subprocess.run(
            ["python", str(script_path), "--config", str(abs_config), "--output_dir", str(output_dir)],
            cwd=None, # Run from root
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Script Execution Timed Out after {timeout} seconds.")
    
    if result.returncode != 0:
        error_output = result.stderr
        if len(error_output) > 1500:
            error_output = "...[TRUNCATED]...\n" + error_output[-1500:]
        raise RuntimeError(f"Script Execution Failed:\n{error_output}")
        
    # Parse final score from metrics.json
    if not metrics_path.exists():
        raise ValueError(f"Could not find metrics.json after script execution. Output was:\n{result.stdout}\n{result.stderr}")
        
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        if "cv_score" not in metrics:
            raise ValueError(f"metrics.json missing 'cv_score' key: {metrics}")
        return float(metrics["cv_score"])
    except json.JSONDecodeError:
        raise ValueError(f"metrics.json is malformed. Content: {metrics_path.read_text()}")

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
    temperature: float = 0.4,
    ollama_base_url: str = None,
    wandb_enabled: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    pred_type: str = "prob",
    config_path: str = "config.yaml",
    available_models: list = None,
    workspace_mgr=None
):
    if available_models is None:
        available_models = []
    log_stage("Starting Agentic Loop")
    
    if wandb_enabled:
        weave_project = wandb_project if wandb_project else "agentic-automl"
        if wandb_entity:
            weave_project = f"{wandb_entity}/{weave_project}"
        weave.init(weave_project)
        
    current_best_score = base_score
    history = []
    
    history_path = Path("history.json")
    # Load history if exists
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
                
            if base_score is None and history:
                higher_is_better = True
                valid_scores = [run['score'] for run in history if run.get('score') is not None]
                if valid_scores:
                    current_best_score = max(valid_scores) if higher_is_better else min(valid_scores)
        except:
            pass

    if current_best_score is None:
        raise ValueError("base_score was None and could not be determined from history.")

    eda_path = "EDA.md"
    eda_content = read_file(eda_path)
    
    start_iteration = len(history) + 1
    end_iteration = start_iteration + max_iterations
    
    failures_in_session = 0
    
    for i in range(start_iteration, end_iteration):
        log_stage(f"Iteration {i}")
        
        # We ensure we are on the dataset branch, but we DO NOT revert changes.
        # This allows a broken script from a failed previous iteration to persist
        # so the LLM can read it and attempt to fix its own errors.
        git_mgr.checkout_branch(dataset_branch)
        
        script_path = "train_model.py"
        current_script = read_file(script_path)
        memory_string = "### Agent Memory (Past Experiments)\n"
        recent_history = history[-3:] if len(history) >= 3 else history

        for run in recent_history:
            status = "IMPROVED" if run.get('improved') else ("FAILED WITH ERROR" if run.get('error') else "DEGRADED")
            memory_string += f"- Iteration {run['iteration']} ({status}): "
            
            if run.get('error'):
                # Truncate error to last 1000 chars
                memory_string += f"{str(run['error'])[-1000:]}\n"
            else:
                memory_string += f"Score {run.get('score')}. Reasoning: {run.get('agent_reasoning', 'No reasoning provided')}\n"

        pred_prob_instruction = "Ensure that for the final `raw_submission.csv`, you ALWAYS predict the continuous PROBABILITIES for the positive class (e.g., using `predict_proba(test_X)[:, 1]`). Do NOT apply any thresholding or class conversion. Another script will handle formatting it for Kaggle into `submission.csv`."

        models_str = ", ".join(available_models) if available_models else "None specifically defined in registry"
        prompt = f"""You are an expert AI Data Scientist. Your goal is to improve the Cross-Validation score of the model.

CRITICAL: Your script MUST remain dataset-agnostic. 
- ALWAYS read `dataset_path`, `target_col`, and `test_path` from the configuration file.
- Support a `--config` command-line argument (using `argparse`) to specify the configuration file path (defaulting to `config.yaml`).
- NEVER hardcode column names (like "Survived") or file paths (like "data/titanic/train.csv").
- Use the `target_col` variable from the config for all target-related operations, including the submission file column name.
- When reading the dataset, you MUST preserve the `nrows=...` argument in `pd.read_csv` to prevent Out-Of-Memory crashes during evaluation.
- ID COLUMN PRESERVATION: When generating `raw_submission.csv`, you MUST capture the original ID column (typically the first column of the test set) before dropping it from the feature set. Failure to do this causes "column shifting" where the ID column is replaced by feature data, leading to invalid submissions.

MODELING FREEDOM: You are NOT restricted to the current model setup (e.g., CatBoost). 
- You are encouraged to change the model architecture, introduce ensembling (using `VotingClassifier`/`Regressor` or `StackingClassifier`/`Regressor`), or try different frameworks (XGBoost, LightGBM, CatBoost, H2O AutoML) to improve the score.
- You can add feature engineering, handle missing values better, and tune hyperparameters.

You have access to the following pre-configured models from the registry: {models_str}. You may tune their hyperparameters, but do not hallucinate imports for models outside of this list unless you are confident they are in the environment.

You should try tuning hyperparameters for these models, comparing their individual performance, or ensembling them (e.g., using `VotingClassifier`/`VotingRegressor` or `StackingClassifier`/`StackingRegressor`) to maximize the cross-validation score.

{memory_string}

YOUR MISSION PRIORITIES:

FIX ERRORS FIRST: If the memory above indicates the previous run failed with a traceback or error, your EXCLUSIVE priority is to debug and fix the script. Do NOT attempt to add new features, models, or optimizations until the error is resolved.

IMPROVE SCORE: If the previous run succeeded, your goal is to propose a modified version of the script to improve the model via feature engineering, missing value handling, hyperparameter tuning, or architecture changes.

Always ensure your script accepts an `--output_dir` command-line argument using `argparse`. You MUST write the final cross-validation score to a file named `metrics.json` and predictions to `raw_submission.csv` inside this `output_dir` using `pathlib.Path(output_dir) / ...`. Do NOT use generic relative paths. The format for metrics should be: `{{"cv_score": final_score}}`.
{pred_prob_instruction}
Output ONLY the full modified Python code wrapped in python ...  blocks. Do not include other text.
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
            
            user_message = {"role": "user", "content": prompt}
            api_key = os.environ.get("GEMINI_API_KEY") if "gemini" in model_name.lower() else os.environ.get("MOONSHOT_API_KEY", os.environ.get("OPENAI_API_KEY"))
            file_messages = get_file_messages(["train_model.py", "EDA.md", config_path], model_name, api_key)
            final_messages = file_messages + [user_message]

            completion_kwargs = {
                "model": model_name,
                "messages": final_messages,
                "temperature": temperature,
                "request_timeout": 120  # Ensure LLM call doesn't hang indefinitely
            }
            # Ollama requires an api_base pointing to the local server
            if model_name.startswith("ollama"):
                base = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                completion_kwargs["api_base"] = base
                
                print(f"  [Info] Local Ollama model detected. Checking connection to {base} ...")
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
            
            llm_output = call_agent_llm(completion_kwargs)
        except (BadRequestError, AuthenticationError) as e:
            log_error("Fatal LLM API Error (Invalid Config or Auth). Terminating loop.", e)
            break
        except Timeout as e:
            log_error("LLM API Call Timed Out", e)
            if "gemini-2.5-flash" not in model_name:
                print(f"  [Warning] Temporary fallback to gemini/gemini-2.5-flash due to timeout.")
                completion_kwargs["model"] = "gemini/gemini-2.5-flash"
                if "api_base" in completion_kwargs:
                    del completion_kwargs["api_base"]
                try:
                    llm_output = call_agent_llm(completion_kwargs)
                except Exception as fallback_e:
                    log_error("Fallback LLM Call also failed", fallback_e)
                    continue
            else:
                continue
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
        
        # Write new code directly to root
        with open("train_model.py", "w") as f:
            f.write(new_code)
            
        try:
            log_stage(f"Evaluating Generated Code")
            new_score = run_training_script(script_path, timeout=timeout, config_path=config_path, workspace_mgr=workspace_mgr)
            log_metric("Iteration Score", new_score)
            
            higher_is_better = True
            
            improved = (new_score > current_best_score) if higher_is_better else (new_score < current_best_score)
            
            if wandb_enabled:
                
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
                    "agent_reasoning": llm_summary
                })
            else:
                log_stage(f"Score degraded or unchanged. Leaving changes uncommitted in workspace for next iteration to retry.")
                history.append({
                    "iteration": len(history)+1,
                    "commit": None,
                    "score": new_score,
                    "improved": False,
                    "agent_reasoning": llm_summary
                })
                
            # Kaggle Submission Logic
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                auto_submit_val = str(config.get("auto_kaggle_submit", "never")).lower()
                
                # We submit if 'always' (or true), OR if 'best' and it improved
                should_submit = False
                if auto_submit_val in ["always", "true"]:
                    should_submit = True
                elif auto_submit_val == "best" and improved:
                    should_submit = True
                    
                if should_submit:

                    raw_sub_path = Path(workspace_mgr.get_file_path("raw_submission.csv")) if workspace_mgr else Path("raw_submission.csv")
                    if raw_sub_path.exists():
                        log_stage(f"Automated Kaggle Submission for Iteration {len(history)}")
                        kaggle_ops.format_submission(config_path, workspace_mgr=workspace_mgr)
                        commit_to_submit = history[-1].get("commit")
                        kaggle_ops.submit_to_kaggle(config_path, commit_id=commit_to_submit, workspace_mgr=workspace_mgr)
            except Exception as e:
                log_error(f"Failed to submit iteration {len(history)} to Kaggle", e)
                
        except Exception as e:
            log_error(f"Execution failed for iteration {i}", e)
            failures_in_session += 1
            history.append({
                "iteration": len(history)+1,
                "commit": None,
                "score": None,
                "improved": False,
                "error": str(e),
                "agent_reasoning": llm_summary if 'llm_summary' in locals() else "Execution failed before logic extraction"
            })
            
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    if max_iterations > 0 and failures_in_session == max_iterations:
        raise RuntimeError(f"All {max_iterations} agent iterations failed during this session. See logs for details.")

    log_stage("Agentic Loop Finished")
    log_metric("Final Best Score", current_best_score)

