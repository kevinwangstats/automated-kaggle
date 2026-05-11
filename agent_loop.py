import os
import json
import re
import subprocess
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

def run_training_script(script_path="train_model.py", timeout: int = 600):
    # Run the script as a subprocess to capture its FINAL_CV_SCORE output safely
    # and prevent crashes in the orchestrator if the generated code is bad.
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Script Execution Timed Out after {timeout} seconds.")
    
    if result.returncode != 0:
        raise RuntimeError(f"Script Execution Failed:\n{result.stderr}")
        
    # Parse final score
    match = re.search(r'FINAL_CV_SCORE: ([\d\.]+)', result.stdout)
    if not match:
        raise ValueError(f"Could not find FINAL_CV_SCORE in output. Output was:\n{result.stdout}")
        
    return float(match.group(1))

def run_agent_loop(
    dataset_path: str,
    target_col: str,
    base_score: float,
    git_mgr: GitManager,
    task: str,
    max_iterations: int = 5,
    skip_confirmation: bool = False,
    timeout: int = 600,
    model: str = None,
    ollama_base_url: str = None
):
    log_stage("Starting Agentic Loop")
    current_best_score = base_score
    history = []
    
    # Load history if exists
    if os.path.exists("history.json"):
        try:
            with open("history.json", "r") as f:
                history = json.load(f)
        except:
            pass

    eda_content = read_file("EDA.md")
    
    for i in range(1, max_iterations + 1):
        log_stage(f"Iteration {i}")
        
        current_script = read_file("train_model.py")
        history_context = ""
        if len(history) > 0:
            last_run = history[-1]
            if last_run.get('error'):
                history_context = f"\\nPREVIOUS RUN FAILED WITH ERROR:\\n{last_run['error']}\\nPlease fix this error.\\n"
            elif not last_run.get('improved'):
                history_context = f"\\nPREVIOUS RUN DEGRADED SCORE ({last_run.get('score')} vs Best: {current_best_score}). Try a different approach.\\n"
            else:
                history_context = f"\\nPREVIOUS RUN IMPROVED SCORE TO {last_run.get('score')}. Good job, keep going!\\n"

        prompt = f"""You are an expert AI Data Scientist. Your goal is to improve the Cross-Validation score of the model.

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
Always ensure you output the FINAL_CV_SCORE in the same format: print(f"FINAL_CV_SCORE: {{final_score:.4f}}")
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
            model_name = model or os.environ.get("AUTOML_MODEL", "gemini/gemini-2.0-flash")
            log_stage(f"Calling LLM: {model_name}")
            
            completion_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4
            }
            # Ollama requires an api_base pointing to the local server
            if model_name.startswith("ollama"):
                base = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                completion_kwargs["api_base"] = base
            
            response = completion(**completion_kwargs)
            llm_output = response.choices[0].message.content
        except Exception as e:
            log_error("LLM API Call failed", e)
            continue
            
        new_code = extract_python_code(llm_output)
        
        # Git Branch creation
        branch_name = git_mgr.create_experiment_branch(iteration=len(history)+1)
        
        # Write new code
        with open("train_model.py", "w") as f:
            f.write(new_code)
            
        try:
            log_stage(f"Evaluating Generated Code")
            new_score = run_training_script("train_model.py", timeout=timeout)
            log_metric("Iteration Score", new_score)
            
            higher_is_better = (task == 'classification')
            
            improved = (new_score > current_best_score) if higher_is_better else (new_score < current_best_score)
            
            if improved:
                log_stage(f"Score improved! ({current_best_score:.4f} -> {new_score:.4f})")
                current_best_score = new_score
                
                # Commit and merge
                commit_id = git_mgr.commit_all(f"[Iter {len(history)+1} | CV Score: {new_score:.4f}] Successful agent iteration")
                git_mgr.merge_to_main(branch_name, f"Merge Iter {len(history)+1}")
                
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
                log_stage(f"Score degraded or unchanged. Discarding branch.")
                git_mgr.discard_branch(branch_name)
                # revert train_model.py to main's version
                git_mgr.checkout_branch("main")
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
            git_mgr.discard_branch(branch_name)
            git_mgr.checkout_branch("main")
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
