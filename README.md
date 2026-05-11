# Agentic AutoML Tabular Pipeline Template

This repository serves as a ready-to-use template for automated, agentic modeling on **tabular** Kaggle data challenges. It implements an LLM-driven pipeline that iteratively improves machine learning model performance on tabular datasets through reasoning and code generation.

## Architecture

The system consists of the following core components:

1. **EDA Engine**: Automatically performs Exploratory Data Analysis on a given dataset and outputs a concise `EDA.md` summary for the LLM context.
2. **Baseline Engine**: Evaluates standard frameworks (XGBoost, LightGBM, CatBoost, H2O) using K-Fold Cross-Validation to establish a baseline model and starting script.
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

To kick off the pipeline, run the `main.py` script and provide the path to your Kaggle `train.csv` dataset:

```bash
python main.py --dataset_path path/to/train.csv --target_col target
```

### Advanced Usage

You can customize the evaluation metric, provide a test dataset for automatic `submission.csv` generation, and enforce time limits on LLM-generated code:

```bash
python main.py \
    --dataset_path path/to/train.csv \
    --target_col target \
    --test_path path/to/test.csv \
    --metric log_loss \
    --timeout 600 \
    -y
```

- `--test_path`: Path to `test.csv`. If provided, the generated scripts will automatically train on the full training set and output predictions for Kaggle submission.
- `--metric`: Override the default metric (defaults are `roc_auc` for classification and `neg_mean_squared_error` for regression).
- `--timeout`: Maximum execution time in seconds for a single LLM-generated script run (default: 600).
- `-y`: Skip the manual user confirmation prompt before each LLM API call.

## Logs and Tracking

- **`CHANGELOG.md`**: Stores a human-readable summary of every successful iteration.
- **`history.json`**: Stores granular metrics, hyperparameters, and git commits for every attempt.
- The pipeline uses a custom token-efficient logger to minimize noise during execution.
