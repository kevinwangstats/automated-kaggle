# Developer Guide: Agentic AutoML

This document contains the technical details, architectural decisions, and development workflows for the Agentic AutoML pipeline. If you are looking for usage instructions, please see [README.md](README.md).

## Architecture

The system consists of the following core components:

1. **EDA Engine**: Automatically performs Exploratory Data Analysis on a given dataset and outputs a concise `EDA.md` summary for the LLM context.
2. **Baseline Engine**: Evaluates standard frameworks (XGBoost, LightGBM, CatBoost) using K-Fold Cross-Validation to establish a baseline model and starting script. For binary classification tasks, it automatically generates detailed out-of-fold metrics including Accuracy, F1 Score, Sensitivity, Specificity, and Positive/Negative case counts. For multi-class tasks, it provides Accuracy, Macro/Micro F1 Scores, and class distribution counts.
3. **Agent Loop**: An orchestrator powered by `litellm` that feeds the dataset context, current code, and performance history to an LLM. The LLM edits the Python training script to improve the cross-validation score. If an iteration fails (due to timeout or error), the broken code is retained in the working directory so the next iteration can attempt to fix it, without polluting the git history.
4. **Git Manager**: Ensures strict provenance. Every experiment runs on a separate `experiment/iter_<N>` branch. Successful iterations (those that beat the current best CV score) are merged into the dataset branch and tracked.

## Git Branches and Provenance

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

When you run the pipeline, it checks out `main`, then creates or checks out the dataset branch. Baseline and agent commits (including merges from successful `experiment/iter_*` runs) go to the **dataset** branch. Failed experiments delete the experiment branch but leave the changes in the working directory for the next iteration to debug.

On a **brand-new** repository with no commits yet, the first baseline commit still creates `main`; the pipeline then creates the dataset branch at the same commit and continues there so later work stays off `main` until you merge intentionally.

## CI/CD and Testing

This template includes a GitHub Actions workflow (`.github/workflows/titanic_ci.yml`) that automatically validates the pipeline on every push to `main` using the classic [Titanic dataset](https://www.kaggle.com/competitions/titanic/overview). It runs the EDA and Baseline engines with `iterations: 0` to skip LLM calls entirely, ensuring the core pipeline is always functional at zero API cost.

The test configuration lives at `tests/titanic_config.yaml` and can also be run **locally**:

```bash
kaggle competitions download -c titanic -p data/titanic && unzip data/titanic/titanic.zip -d data/titanic/
python main.py --config tests/titanic_config.yaml -y
```

**Required GitHub Repository Secret:**

- `KAGGLE_API_TOKEN`: Your Kaggle access token (generate at kaggle.com > Account Settings > API).
