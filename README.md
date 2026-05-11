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

2. Set your preferred LLM API keys in your environment variables. For example:

   ```bash
   export OPENAI_API_KEY="sk-..."
   export GEMINI_API_KEY="AIza..."
   ```

## Usage

This template now uses a YAML configuration file as the single source of truth for your pipeline inputs, providing a cleaner level of abstraction.

1. Open `config.yaml` and configure your paths and metrics:
```yaml
dataset_path: "data/train.csv"
target_col: "target"
test_path: "data/test.csv"
metric: "log_loss"
iterations: 5
timeout: 600
```

2. To kick off the pipeline, simply run the orchestrator:

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

## Logs and Tracking

- **`CHANGELOG.md`**: Stores a human-readable summary of every successful iteration.
- **`history.json`**: Stores granular metrics, hyperparameters, and git commits for every attempt.
- The pipeline uses a custom token-efficient logger to minimize noise during execution.

## CI/CD

This template includes a GitHub Actions workflow (`.github/workflows/titanic_ci.yml`) that automatically validates the pipeline on every push to `main` using the classic [Titanic dataset](https://www.kaggle.com/competitions/titanic/overview). It runs the EDA and Baseline engines with `iterations: 0` to skip LLM calls entirely, ensuring the core pipeline is always functional at zero API cost.

**Required GitHub Repository Secret:**
- `KAGGLE_API_TOKEN`: Your Kaggle access token (generate at kaggle.com > Account Settings > API).
