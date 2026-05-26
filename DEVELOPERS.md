# Developer Guide: Agentic AutoML

This document contains the technical details, architectural decisions, and development workflows for the Agentic AutoML pipeline. If you are looking for usage instructions, please see [README.md](README.md).

## Architecture

The system consists of the following core components:

1. **EDA Engine**: Automatically performs Exploratory Data Analysis on a given dataset and outputs a concise `EDA.md` summary for the LLM context.
2. **Baseline Engine**: Evaluates standard frameworks (XGBoost, LightGBM, CatBoost, Ridge Regression) via K-Fold CV to establish a baseline. It generates an initial, ephemeral `train_model.py` ensemble script that is continuously overwritten (unless resuming). It also automatically logs out-of-fold classification metrics and distributions.
3. **Agent Loop**: An orchestrator powered by `litellm`. It uses a **Multi-Message architecture with local text reading and smart truncation** (truncating large files like `EDA.md` to keep context windows lightweight, while keeping `train_model.py` fully intact) to inject relevant context directly into the prompt. To prevent network and gateway timeouts (especially critical for Kimi/Moonshot), the LLM connection leverages native **streaming** to receive responses incrementally. The LLM edits the Python training script directly on the dataset branch. Failed iterations retain the broken code so the agent can debug itself. **To prevent context window bloat and reduce TTFT (Time-to-First-Token), script tracebacks are aggressively truncated (max 1500 chars) before being fed back to the LLM.** Successful iterations are committed and optionally submitted to Kaggle.
4. **Git Manager**: Ensures strict provenance. All dataset-specific work runs on a dedicated dataset branch. Successful iterations are directly committed and tracked.
5. **Workspace Manager**: Isolates all ephemeral artifacts (`train_model.py`, `metrics.json`, `history.json`, `EDA.md`, `raw_submission.csv`) into a static `.workspaces/<dataset_branch>/` directory. This keeps the repository root clean and enables parallel experiments on different datasets without file collisions.
6. **Resume & Continuity**: The pipeline detects previous state in the workspace and prompts the user to resume or start from scratch. The `run_mode` config key (`"prompt"`, `"resume"`, `"scratch"`) controls this behavior. When resuming, the orchestrator **always executes the existing `train_model.py` as-is** to establish a ground-truth baseline score. It parses `history.json` and compares the last known LLM-generated code against the current script; if manual modifications are detected, a `HUMAN_INTERVENTION` entry is automatically injected into `history.json` using the newly evaluated empirical score. This preserves strict code provenance and guarantees the agent always iterates on verified metrics. It also reloads the W&B run ID from `wandb_run_id.txt` to ensure continuity.
7. **Observability & Logging**: The pipeline unifies console output through `logger.py` (`log_info`, `log_stage`, `log_error`), preventing inconsistent outputs. It supports optionally writing this stream to a local file via `--log-file`. Additionally, if Weights & Biases is enabled, `agent_loop.py` uses `@weave.op()` to trace and version all LLM inputs, configurations, and outputs securely.

## Agentic Memory & API Defense
To maximize API efficiency, prevent infinite loops, and reduce token bloat, the pipeline implements the following safeguards:
- **Semantic Memory**: The orchestrator no longer passes raw code history (prompts and responses) into the LLM context. Instead, it maintains a distilled "Memory String" summarizing only the outcomes and the agent's extracted reasoning from the last 3 runs. This saves significant tokens and forces the agent to maintain focus on high-level strategies rather than drowning in thousands of lines of previous code.
- **API Defense (Transient vs. Fatal)**: API calls are wrapped in robust litellm exception handlers. **Fatal errors** (e.g., 400 Bad Request, 401 Authentication Error) immediately halt the pipeline to prevent infinite loops and save costs. **Transient Timeouts** trigger an automatic, temporary fallback to `gemini-2.5-flash` to ensure the pipeline continues iterating smoothly.

### Dual-Mode Cognitive State Machine
To prevent LLM attention dilution, the agent orchestrator utilizes a state machine. If the script is stable, it enters Optimize Mode, focusing entirely on feature engineering and hyperparameter tuning. If the script crashes, it enters a strict Debug Mode. In Debug Mode, the prompt explicitly forbids adding new features and mandates that the agent must repair the broken logic rather than taking the lazy route of deleting it.

## Agentic Memory & API Defense
To maximize API efficiency, prevent infinite loops, and reduce token bloat, the pipeline implements the following safeguards:
- **Semantic Memory**: The orchestrator no longer passes raw code history (prompts and responses) into the LLM context. Instead, it maintains a distilled "Memory String" summarizing only the outcomes and the agent's extracted reasoning from the last 3 runs. This saves significant tokens and forces the agent to maintain focus on high-level strategies rather than drowning in thousands of lines of previous code.
- **API Defense (Transient vs. Fatal)**: API calls are wrapped in robust litellm exception handlers. **Fatal errors** (e.g., 400 Bad Request, 401 Authentication Error) immediately halt the pipeline to prevent infinite loops and save costs. **Transient Timeouts** trigger an automatic, temporary fallback to `gemini-2.5-flash` to ensure the pipeline continues iterating smoothly.

## Git Branches and Worktree Architecture

This project enforces a strictly data-agnostic workflow using **Git Worktrees**. The core codebase remains on `main` in the primary repository directory, while data-specific iterations and configurations are isolated to dataset branches checked out into parallel sibling directories.

### Physical Directory Layout

```text
/your-projects/
├── automated-kaggle/             [Branch: main] Primary Repo (Core Engine)
│   ├── main.py
│   ├── agent_loop.py
│   ├── baseline_engine.py
│   ├── workspace_manager.py
│   └── .workspaces/              (git-ignored, ephemeral artifacts)
│       └── titanic/
│           ├── train_model.py
│           ├── history.json
│           ├── EDA.md
│           └── metrics.json
│
├── automated-kaggle-titanic/     [Branch: titanic] Worktree (Dataset Experiment)
│   ├── config.yaml
│   └── .workspaces/titanic/
│       ├── train_model.py
│       └── history.json
│
└── automated-kaggle-spaceship/   [Branch: spaceship] Worktree (Dataset Experiment)
    ├── config.yaml
    └── .workspaces/spaceship/
        └── ...
```

### Working with New Datasets

To start a new competition or dataset experiment without cluttering the `main` branch or switching context manually:

1.  **Create a new branch** from `main`:
    ```bash
    git branch my-new-dataset
    ```

2.  **Initialize a new Worktree** in a sibling folder:
    ```bash
    git worktree add ../automated-kaggle-my-new-dataset my-new-dataset
    ```

3.  **Setup and Run**:
    ```bash
    cd ../automated-kaggle-my-new-dataset
    # Add your data files to the local data/ folder (git-ignored)
    # Edit config.yaml for this specific dataset
    python main.py
    ```

4.  **Sync with Core Engine**:
    If updates are made to the core engine on the `main` branch, merge them into your dataset worktree:
    ```bash
    # Inside the dataset worktree directory:
    git merge main
    ```

- **`main`**: Shared **backbone** (orchestration, engines, generic defaults).
- **Dataset Worktree**: Isolated environment for a specific competition. All generated files (`train_model.py`, `EDA.md`, `history.json`) stay here, safely separated from the core engine.

When the pipeline runs within a worktree, the `git_manager.py` respects the current branch context. All agent iterations are committed directly to the dataset branch within that worktree.


## CI/CD and Testing

This template includes a GitHub Actions workflow (`.github/workflows/titanic_ci.yml`) that automatically validates the pipeline on every push to `main` using the classic [Titanic dataset](https://www.kaggle.com/competitions/titanic/overview). It runs the EDA and Baseline engines with `iterations: 0` to skip LLM calls entirely, ensuring the core pipeline is always functional at zero API cost.

The test configuration lives at `tests/titanic_config.yaml` and can also be run **locally**:

```bash
kaggle competitions download -c titanic -p data/titanic && unzip data/titanic/titanic.zip -d data/titanic/
python main.py --config tests/titanic_config.yaml -y
```

**Required GitHub Repository Secret:**

- `KAGGLE_API_TOKEN`: Your Kaggle access token (generate at kaggle.com > Account Settings > API).
