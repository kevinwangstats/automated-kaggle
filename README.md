# Agentic AutoML Tabular Pipeline Template

This repository serves as a ready-to-use template for automated, agentic modeling on **tabular** Kaggle data challenges. It implements an LLM-driven pipeline that iteratively improves machine learning performance by concurrently optimizing and ensembling a suite of powerful frameworks (e.g., **XGBoost**, **LightGBM**, **CatBoost**, **HistGradientBoosting**), while also supporting evaluation of **H2O AutoML** baselines.

The pipeline establishes a baseline ensemble out of the box, and then uses LLM-driven reasoning to tune hyperparameters, perform feature engineering, and refine the ensemble architecture to maximize cross-validation scores.

If you are looking for technical details on how the pipeline is built, its architecture, or how to contribute, please see [DEVELOPERS.md](DEVELOPERS.md).

## Setup Instructions

1. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Isolation with Git Worktree.** This project uses a parallel directory structure to isolate the core engine (`main` branch) from dataset-specific artifacts (e.g., `titanic` branch). This prevents branch-switching friction and keeps your workspace clean.

   To set up a new dataset environment, simply run:

   ```bash
   python kaggle_ops.py --setup <dataset_name>
   ```
   
   This will automatically install dependencies, branch the repository, create an isolated worktree, and download/extract the competition data via the Kaggle API.

3. **Configure your LLM.** This pipeline uses [`litellm`](https://docs.litellm.ai/docs/providers) to support any LLM provider. Model selection follows this priority order:
   1. `model` field in `config.yaml` (highest priority)
   2. `AUTOML_MODEL` environment variable
   3. Default: `gemini/gemini-2.5-flash-lite`

   **Gemini (default):**

   ```bash
   export GEMINI_API_KEY="AIza..."
   # config.yaml: model: gemini/gemini-2.0-flash
   ```

   **OpenAI / ChatGPT:**

   ```bash
   export OPENAI_API_KEY="sk-..."
   # config.yaml: model: openai/gpt-4o
   ```

   **Anthropic:**

   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   # config.yaml: model: anthropic/claude-3-5-sonnet-20241022
   ```

   **Ollama (local, no API key required):**

   ```bash
   # 1. Install Ollama: https://ollama.com
   # 2. Pull a model:
   ollama pull llama3.2
   # 3. Set in config.yaml:
   #    model: ollama/llama3.2
   #    ollama_base_url: http://localhost:11434  (optional, this is the default)
   ```

   See the full list of supported providers at [docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers).

## Usage

This template uses a YAML configuration file as the single source of truth for your pipeline inputs.

1. Open `config.yaml` and configure your paths and metrics. Use a per-competition folder under `data/` (for example `data/titanic/train.csv`).

```yaml
dataset_path: "data/train.csv"
target_col: "target"
test_path: "data/test.csv"
metric: "log_loss"
iterations: 5
timeout: 600
```

1. To kick off the pipeline, simply run the orchestrator:

```bash
python main.py
```

### Advanced Usage

You can override the default config file path, skip user confirmation, or resume a previous run:

```bash
python main.py --config custom_config.yaml -y --resume
```

- `--config`: Path to a custom YAML configuration file (default: `config.yaml`).
- `-y`: Skip the manual user confirmation prompt before each LLM API call. When previous state is detected, this flag skips the interactive resume/restart prompt and falls back to the `run_mode` defined in `config.yaml` (default: `resume`).
- `-r`, `--resume`: Force-resume an existing optimization session. This skips the EDA and Baseline generation phases, loading the previous best score and W&B run ID to continue iterating directly from your current best `train_model.py` script.

### Resume or Restart Safety Check

If the pipeline detects a previous `train_model.py` and `history.json` in the workspace, it will prompt you interactively:

```
[Warning] Previous iterations detected for this dataset. Do you want to resume from the existing train_model.py and history? (y/n):
```

This prevents accidentally overwriting manually optimized scripts. You can control this behavior via the `run_mode` key in `config.yaml`:

| `run_mode` | Behavior |
|---|---|
| `"prompt"` | Ask interactively (default) |
| `"resume"` | Always resume from existing state |
| `"scratch"` | Always start fresh |

**Important Note on Resuming:**
When you resume an optimization session, the pipeline will **always execute your existing `train_model.py` as-is first** to establish an empirical cross-validation score. If you manually edit the script between runs, this guarantees your changes are properly evaluated, and the pipeline will automatically log a `HUMAN_INTERVENTION` entry in `history.json` featuring the updated code and its newly evaluated score. This ensures the LLM is always trying to beat a true, verified baseline.

## Outputs and Tracking

All pipeline artifacts are isolated inside a static workspace directory at `.workspaces/<dataset_branch>/` (e.g., `.workspaces/titanic/`). This keeps the repository root clean and enables parallel experiments on different datasets.

- **`train_model.py`**: The current best Python script generated by the pipeline. Note that this file is **ephemeral** and will be automatically overwritten by `baseline_engine.py` at the start of each new run (unless resuming).
- **`EDA.md`**: An automatically generated exploratory data analysis summary.
- **`metrics.json`**: The cross-validation score contract between the generated script and the orchestrator.
- **`CHANGELOG.md`**: Stores a human-readable summary of every successful iteration.
- **`history.json`**: Stores granular metrics, hyperparameters, and git commits for every attempt.
- **Kaggle Submissions**: The `auto_kaggle_submit` setting in `config.yaml` controls automated API submissions. If set to `"always"`, the pipeline submits to Kaggle on every successful execution that generates a valid prediction file. If set to `"best"`, it submits only when the local CV score strictly beats the historical best. The automated submission message dynamically includes the short Git commit SHA ID of the codebase that generated it, ensuring precise code provenance.
