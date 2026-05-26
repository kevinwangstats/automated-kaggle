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
import time
import re
import subprocess
import yaml
import sys
import urllib.request
import urllib.error
import wandb
import weave
import kaggle_ops
from pathlib import Path
from litellm import completion
from litellm.exceptions import Timeout, BadRequestError, AuthenticationError
from logger import log_stage, log_error, log_metric, log_info
from git_manager import GitManager

from utils import read_file

def get_file_messages(file_paths: list) -> list:
    """Reads files locally and formats them as system messages with smart truncation."""
    messages = []
    MAX_EDA_CHARS = 8000
    
    log_info("Reading context files locally...")
    for fp in file_paths:
        if not Path(fp).exists():
            continue
        content = read_file(fp)
        
        # Truncate large EDA files to preserve LLM context window.
        # Never truncate train_model.py — the LLM needs the full code.
        if "EDA" in fp and len(content) > MAX_EDA_CHARS:
            content = content[:MAX_EDA_CHARS] + "\n... [TRUNCATED FOR BREVITY] ..."
        
        messages.append({"role": "system", "content": f"--- {fp} ---\n{content}"})
                
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

@weave.op()
def call_agent_llm(completion_kwargs: dict) -> str:
    """Wrapper for LLM completion with streaming to prevent gateway timeouts."""
    completion_kwargs["stream"] = True
    response = completion(**completion_kwargs)
    
    chunks = []
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
    
    return "".join(chunks)

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
            [sys.executable, str(script_path), "--config", str(abs_config), "--output_dir", str(output_dir)],
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
        script_content = read_file(script_path) if Path(script_path).exists() else "File not found."
        raise ValueError(f"Could not find metrics.json after script execution. The script may not have executed correctly or did not write to metrics.json.\n\nScript content was:\n{script_content}\n\nOutput was:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        if "cv_score" not in metrics:
            raise ValueError(f"metrics.json missing 'cv_score' key: {metrics}")
        return float(metrics["cv_score"])
    except json.JSONDecodeError:
        raise ValueError(f"metrics.json is malformed. Content: {metrics_path.read_text()}")

def _init_weave(wandb_enabled: bool, wandb_project: str, wandb_entity: str):
    if wandb_enabled:
        weave_project = wandb_project if wandb_project else "agentic-automl"
        if wandb_entity:
            weave_project = f"{wandb_entity}/{weave_project}"
        weave.init(weave_project)

def _build_agent_memory(history: list) -> str:
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
            
    return memory_string

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
    workspace_mgr=None,
    strict_mode: bool = False,
    ci_test_mode: bool = False
):
    agent_loop_start_time = time.time()
    if available_models is None:
        available_models = []
    log_stage("Starting Agentic Loop")
    
    _init_weave(wandb_enabled, wandb_project, wandb_entity)
        
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
        iter_start_time = time.time()
        
        # We ensure we are on the dataset branch, but we DO NOT revert changes.
        # This allows a broken script from a failed previous iteration to persist
        # so the LLM can read it and attempt to fix its own errors.
        git_mgr.checkout_branch(dataset_branch)
        
        script_path = "train_model.py"
        current_script = read_file(script_path)
        memory_string = _build_agent_memory(history)

        # Determine Cognitive State
        last_run_failed = False
        if history:
            last_run = history[-1]
            if last_run.get('error'):
                error_str = str(last_run['error'])
                # Only trigger strict Debug Mode for script execution failures, not LLM API timeouts
                if "Script Execution Failed" in error_str or "Traceback" in error_str or "SyntaxError" in error_str:
                    last_run_failed = True

        models_str = ", ".join(available_models) if available_models else "None specifically defined in registry"
        
        base_prompt = f"""You are an expert AI Data Scientist. Improve the Cross-Validation score of the model.

RULES (your script MUST follow ALL of these):
1. DATASET-AGNOSTIC: Read `dataset_path`, `target_col`, `test_path` from config. Never hardcode column names or file paths.
2. CLI INTERFACE: Accept `--config` (default `config.yaml`) and `--output_dir` (default `.`) via `argparse`.
3. OUTPUT: Save `metrics.json` (format: `{{"cv_score": final_score}}`) and `raw_submission.csv` inside `output_dir` using `pathlib.Path(output_dir) / ...`.
4. MEMORY SAFETY: Preserve any `nrows=...` argument in `pd.read_csv` to prevent OOM crashes.
5. ID COLUMN: Capture the test set's first column (ID) before dropping it from features. Failure causes column shifting in submissions.
6. PREDICTIONS: For `raw_submission.csv`, always output continuous probabilities via `predict_proba(test_X)[:, 1]`. No thresholding — another script handles Kaggle formatting.

MODELING FREEDOM:
- You may change the model, introduce ensembling (`VotingClassifier`/`StackingClassifier`), or switch frameworks (XGBoost, LightGBM, CatBoost).
- You can add feature engineering, improve missing value handling, and tune hyperparameters.
- Available registry models: {models_str}. You may tune their hyperparameters but do not hallucinate imports for models outside this list unless confident they are installed.

{memory_string}
"""

        if last_run_failed:
            mission_prompt = f"""MISSION (DEBUG MODE - CRITICAL):
The previous execution crashed with the error shown in the memory above.
Your EXCLUSIVE priority is to debug and fix the script so it executes successfully.

DO NOT attempt to add new features, swap models, or optimize the score in this turn.

DO NOT take the "lazy fix" by simply deleting the lines of code that caused the error. You must logically repair the code (e.g., add .astype(str), .fillna(), or correct the matrix dimensions) so the intended feature or logic works.
Output ONLY the full fixed Python code wrapped in ```python ... ``` blocks. No other text.
"""
        else:
            mission_prompt = f"""MISSION (OPTIMIZE MODE):
The previous run succeeded. Your goal is to propose a modified version of the script to improve the Cross-Validation score.

You may perform feature engineering, handle missing values better, tune hyperparameters, or change the ensemble architecture.
Output ONLY the full modified Python code wrapped in ```python ... ``` blocks. No other text.
"""
        
        prompt = base_prompt + "\n" + mission_prompt
        
        if not skip_confirmation:
            log_info(f"Preparing to call LLM for iteration {i}.")
            confirm = input("Continue with this API call? (y/n): ")
            if confirm.lower() != 'y':
                log_info("Skipping LLM call and aborting loop.")
                break
                
        # Call LLM
        try:
            # Priority: config.yaml 'model' > AUTOML_MODEL env var > default
            model_name = model or os.environ.get("AUTOML_MODEL", "gemini/gemini-2.5-flash-lite")
            log_stage(f"Calling LLM: {model_name}")
            
            user_message = {"role": "user", "content": prompt}
            file_messages = get_file_messages(["train_model.py", "EDA.md"])
            final_messages = file_messages + [user_message]

            completion_kwargs = {
                "model": model_name,
                "messages": final_messages,
                "temperature": temperature,
                "request_timeout": 600  # Long timeout to accommodate streaming code generation
            }
            # Ollama requires an api_base pointing to the local server
            if model_name.startswith("ollama"):
                base = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                completion_kwargs["api_base"] = base
                
                log_info(f"Local Ollama model detected. Checking connection to {base} ...")
                try:
                    urllib.request.urlopen(base, timeout=3)
                except urllib.error.URLError:
                    log_info(f"Could not connect to Ollama at {base}.")
                    log_info(f"Please ensure the Ollama app is running locally.")
                log_info(f"Waiting for Ollama response... (this may take a while depending on your hardware)")
            else:
                provider = model_name.split('/')[0] if '/' in model_name else model_name
                log_info(f"Using remote API ({provider}). Please ensure your {provider.upper()}_API_KEY is set if you haven't.")
                log_info(f"Waiting for API response...")
            
            llm_output = call_agent_llm(completion_kwargs)
        except (BadRequestError, AuthenticationError) as e:
            log_error("Fatal LLM API Error (Invalid Config or Auth). Terminating loop.", e)
            break
        except Timeout as e:
            log_error("LLM API Call Timed Out", e)
            if "gemini-2.5-flash" not in model_name:
                log_info(f"Temporary fallback to gemini/gemini-2.5-flash due to timeout.")
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
            log_error("LLM API Call failed (Timeout or other error)", e)
            if "ollama" in (model_name or "").lower():
                log_info(f"If you haven't pulled this model yet, open a new terminal and run: ollama pull {model_name.replace('ollama/', '')}")
            continue
            
        try:
            new_code = extract_python_code(llm_output)
            
            # Extract summary/reasoning for W&B
            llm_summary = re.sub(r'```python.*?```', '', llm_output, flags=re.DOTALL).strip()
            if not llm_summary:
                llm_summary = "No reasoning provided by LLM."
            
            # Write new code directly to root
            target_script = "train_model.py" if not ci_test_mode else "train_model_ci_test.py"
            with open(target_script, "w") as f:
                f.write(new_code)
        except Exception as e:
            log_error(f"Failed to extract or write code for iteration {i}", e)
            failures_in_session += 1
            history.append({
                "iteration": len(history)+1,
                "commit": None,
                "score": None,
                "improved": False,
                "error": f"Extraction/Write Failed: {e}",
                "agent_reasoning": "Extraction/Write Failed"
            })
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            log_info(f"Iteration {i} completed in {time.time() - iter_start_time:.2f} seconds.")
            continue
            
        try:
            if ci_test_mode:
                log_stage("CI Test Mode Active: Bypassing code execution")
                new_score = current_best_score + 0.0001 if current_best_score is not None else 0.9999
            else:
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
            
        log_info(f"Iteration {i} completed in {time.time() - iter_start_time:.2f} seconds.")

    if max_iterations > 0 and failures_in_session == max_iterations:
        if strict_mode:
            raise RuntimeError(f"All {max_iterations} agent iterations failed during this session. See logs for details.")
        else:
            log_stage("WARNING: All agent iterations failed during this session. Continuing pipeline gracefully.")

    if max_iterations > 0 and failures_in_session == max_iterations:
        if strict_mode:
            raise RuntimeError(f"All {max_iterations} agent iterations failed during this session. See logs for details.")
        else:
            log_stage("WARNING: All agent iterations failed during this session. Continuing pipeline gracefully.")

    log_stage("Agentic Loop Finished")
    log_info(f"Total time used for agent loop: {time.time() - agent_loop_start_time:.2f} seconds.")
    log_metric("Final Best Score", current_best_score)

