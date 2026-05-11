# Agentic AutoML Pipeline Skills

This file contains knowledge and skills specific to how this project operates. Agents reading this repository should consult these mechanics when making structural changes.

## 1. How the Agentic Loop Works
- The entry point is `main.py`.
- It triggers `eda_engine.py` to generate `EDA.md`.
- `baseline_engine.py` runs basic models (XGBoost, LightGBM, CatBoost) via K-Fold CV, determines the best performer, and writes the *initial* `train_model.py` template.
- `agent_loop.py` takes control. It reads `EDA.md` and the current `train_model.py`.
- The agent loop calls `litellm` via `gemini/gemini-1.5-pro` (or the configured environment variable) to rewrite `train_model.py`.
- The new script is executed in an isolated `subprocess.run` with a strict timeout.
- If the script fails, the `RuntimeError` or `TimeoutExpired` is captured and fed *back* into the next iteration's prompt.
- If the CV score improves, the git branch is merged to main.

## 2. Metric Maximization / Minimization
- The pipeline delegates metric scoring to standard `scikit-learn` metrics.
- Scikit-learn normalizes regression metrics to be "negative" (e.g., `neg_mean_squared_error`). Therefore, mathematically, a *higher* score is always better (closer to 0).
- The `agent_loop.py` explicitly relies on this fact: `higher_is_better` is generally treated as `True` for raw sklearn scores.

## 3. Submission Generation
- If the `--test_path` argument is provided to `main.py`, the `baseline_engine.py` appends a "submission generation" block to the bottom of the generated `train_model.py`.
- This block fits the final model on `X` and `y` (the entire train set), reads the test set, matches categorical encoding, drops missing columns, and generates a `submission.csv`.
- The LLM is instructed to maintain this submission logic if it rewrites the script.

## 4. Environment Variables
- `AUTOML_MODEL`: Determines the Litellm routing model (e.g. `gemini/gemini-1.5-pro`).
- Standard API keys (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`) must be present in the shell environment running the orchestrator.
