# Agentic AutoML Pipeline Skills

This file contains knowledge and skills specific to how this project operates. Agents reading this repository should consult these mechanics when making structural changes.

## 1. How the Agentic Loop Works
- The entry point is `main.py`.
- It triggers `eda_engine.py` to generate `EDA.md`.
- `baseline_engine.py` runs basic models (XGBoost, LightGBM, CatBoost, H2O AutoML) via K-Fold CV, determines the best performer, and writes the *initial* `train_model.py` template.
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
- If `test_path` is set in `config.yaml`, the `baseline_engine.py` appends a "submission generation" block to the bottom of the generated `train_model.py`.
- This block fits the final model on `X` and `y` (the entire train set), reads the test set, matches categorical encoding, fills missing columns with 0, and generates a `submission.csv`.
- The LLM is instructed to maintain this submission logic if it rewrites the script.

## 4. Configuration
- All pipeline inputs (`dataset_path`, `target_col`, `test_path`, `metric`, `iterations`, `timeout`) are defined in `config.yaml`.
- `main.py` accepts a single `--config` argument pointing to a YAML file (default: `config.yaml`).
- The `-y` flag skips per-LLM-call confirmation prompts.

## 5. Baseline Metrics Reporting
- For binary classification tasks, `baseline_engine.py` runs `cross_val_predict` in addition to `cross_val_score` for all four models (XGBoost, LightGBM, CatBoost, H2O AutoML).
- It computes a Confusion Matrix and derives Accuracy, F1 Score, Sensitivity, and Specificity.
- These reports are printed directly to standard output *outside* the `suppress_stdout_stderr()` block, ensuring they appear in both local and GitHub Actions logs.
- LightGBM models must always be initialized with `verbose=-1` to suppress C++ backend warnings at the source.

## 6. CI/CD Pipeline
- `.github/workflows/titanic_ci.yml` is a GitHub Actions workflow that triggers on every push/PR to `main`.
- It authenticates to Kaggle using a `KAGGLE_API_TOKEN` secret (stored in GitHub > Settings > Secrets), writes it to `~/.kaggle/kaggle.json`, and downloads the Titanic competition dataset.
- It then runs `main.py` with `iterations: 0` to test the EDA and Baseline engines without triggering any LLM calls.
- This makes the Titanic competition a permanent CI regression test for the template.

## 7. Environment Variables
- `AUTOML_MODEL`: Determines the Litellm routing model (e.g. `gemini/gemini-1.5-pro`).
- Standard API keys (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`) must be present in the shell environment running the orchestrator.
- `KAGGLE_API_TOKEN`: Required for the GitHub Actions CI/CD workflow to authenticate to Kaggle.
