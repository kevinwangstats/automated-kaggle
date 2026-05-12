# Agentic AutoML Tabular Pipeline Template

This repository serves as a ready-to-use template for automated, agentic modeling on **tabular** Kaggle data challenges. It implements an LLM-driven pipeline that iteratively improves machine learning model performance on tabular datasets through reasoning and code generation.

## Architecture

The system consists of the following core components:

1. **EDA Engine**: Automatically performs Exploratory Data Analysis on a given dataset and outputs a concise `EDA.md` summary for the LLM context.
2. **Baseline Engine**: Evaluates standard frameworks (XGBoost, LightGBM, CatBoost) using K-Fold Cross-Validation to establish a baseline model and starting script. For binary classification tasks, it automatically generates detailed out-of-fold metrics including Accuracy, F1 Score, Sensitivity, Specificity, and Positive/Negative case counts.
3. **Agent Loop**: An orchestrator powered by `litellm` that feeds the dataset context, current code, and performance history to an LLM. The LLM edits the Python training script to improve the cross-validation score.
4. **Git Manager**: Ensures strict provenance. Every experiment runs on a separate `experiment/iter_<N>` branch. Successful iterations (those that beat the current best CV score) are merged into `main` and tracked.

## Setup Instructions

1. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your LLM.** This pipeline uses [`litellm`](https://docs.litellm.ai/docs/providers) to support any LLM provider. Model selection follows this priority order:
   1. `model` field in `config.yaml` (highest priority)
   2. `AUTOML_MODEL` environment variable
   3. Default: `gemini/gemini-2.0-flash`

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

This template now uses a YAML configuration file as the single source of truth for your pipeline inputs, providing a cleaner level of abstraction.

1. Open `config.yaml` and configure your paths and metrics. Use a per-competition folder under `data/` (for example `data/titanic/train.csv`) so Git work lands on a matching branch name (see [Git branches](#git-branches)).

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

You can override the default config file path or skip the manual user confirmation prompt before each LLM API call:

```bash
python main.py --config custom_config.yaml -y
```

- `--config`: Path to a custom YAML configuration file (default: `config.yaml`).
- `-y`: Skip the manual user confirmation prompt before each LLM API call.

## Git branches

This project enforces a strictly data-agnostic workflow. The core codebase remains on `main`, while data-specific iterations and configurations are isolated to dataset branches. All actual dataset files are ignored by git.

Here is the structural mapping of how files and branches interact:

```text
automated-kaggle/
├── [Branch: main] Core Files (Project-Agnostic)
│   ├── main.py
│   ├── config.yaml (default template)
│   ├── agent_loop.py
│   ├── baseline_engine.py
│   ├── eda_engine.py
│   ├── git_manager.py
│   └── logger.py
│
├── [Branch: <dataset>] Data-Specific Output (Derived from dataset folder)
│   ├── train_model.py (generated)
│   ├── EDA.md (generated)
│   ├── config.yaml (dataset-specific tweaks)
│   ├── history.json
│   └── CHANGELOG.md
│
└── data/ (Git-ignored entirely except .gitkeep)
    ├── .gitkeep
    ├── titanic/          --> Maps to branch "titanic"
    │   ├── train.csv     (Untracked)
    │   └── test.csv      (Untracked)
    └── my-competition/   --> Maps to branch "my-competition"
        └── train.csv     (Untracked)
```

- **`main`**: Keep shared **backbone** here (orchestration, engines, generic defaults). Avoid landing competition-specific artifacts on `main` when you can keep them on a dataset branch instead.
- **Dataset branch**: Derived from `dataset_path` in `config.yaml`. Examples:
  - `data/titanic/train.csv` → branch **`titanic`**
  - `data/my-competition/train.csv` → **`my-competition`**
  - `data/train.csv` (file directly under `data/`) → branch named from the file stem, e.g. **`train`**

When you run the pipeline, it checks out `main`, then creates or checks out the dataset branch. Baseline and agent commits (including merges from successful `experiment/iter_*` runs) go to the **dataset** branch. Failed experiments delete the experiment branch and return to the dataset branch.

On a **brand-new** repository with no commits yet, the first baseline commit still creates `main`; the pipeline then creates the dataset branch at the same commit and continues there so later work stays off `main` until you merge intentionally.

## Logs and Tracking

- **`CHANGELOG.md`**: Stores a human-readable summary of every successful iteration.
- **`history.json`**: Stores granular metrics, hyperparameters, and git commits for every attempt.
- The pipeline uses a custom token-efficient logger to minimize noise during execution.

## CI/CD

This template includes a GitHub Actions workflow (`.github/workflows/titanic_ci.yml`) that automatically validates the pipeline on every push to `main` using the classic [Titanic dataset](https://www.kaggle.com/competitions/titanic/overview). It runs the EDA and Baseline engines with `iterations: 0` to skip LLM calls entirely, ensuring the core pipeline is always functional at zero API cost.

The test configuration lives at `tests/titanic_config.yaml` and can also be run **locally**:

```bash
kaggle competitions download -c titanic -p data/titanic && unzip data/titanic/titanic.zip -d data/titanic/
python main.py --config tests/titanic_config.yaml -y
```

**Required GitHub Repository Secret:**

- `KAGGLE_API_TOKEN`: Your Kaggle access token (generate at kaggle.com > Account Settings > API).
