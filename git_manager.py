import git
import os
import re
from pathlib import Path

from logger import log_stage, log_error


def sanitize_git_branch(name: str) -> str:
    """Map a folder or slug to a valid git branch name; avoid colliding with main."""
    name = (name or "").strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.")
    if not name or name == "main":
        return "default"
    return name


def dataset_branch_from_dataset_path(dataset_path: str) -> str:
    """
    Derive the dataset work branch from config's dataset_path.
    e.g. data/titanic/train.csv -> titanic; data/train.csv -> train (stem).
    """
    parts = Path(dataset_path).parts
    if len(parts) >= 3 and parts[0] == "data":
        return sanitize_git_branch(parts[1])
    if len(parts) == 2 and parts[0] == "data":
        return sanitize_git_branch(Path(dataset_path).stem)
    parent = Path(dataset_path).parent
    if parent.name and parent.name not in (".", "", "/"):
        return sanitize_git_branch(parent.name)
    return sanitize_git_branch(Path(dataset_path).stem)


class GitManager:
    def __init__(self, repo_path="."):
        self.repo_path = os.path.abspath(repo_path)
        try:
            self.repo = git.Repo(self.repo_path)
        except git.exc.InvalidGitRepositoryError:
            self.repo = git.Repo.init(self.repo_path)
            log_stage("Initialized new Git repository.")
            
        self._ensure_main_branch()

    def _ensure_main_branch(self):
        """If HEAD is detached but main exists, attach to main. Never force-switch off a topic branch."""
        if not self.repo.heads:
            return
        try:
            self.repo.active_branch
        except TypeError:
            if "main" in [h.name for h in self.repo.heads]:
                try:
                    self.repo.git.checkout("main")
                except Exception as e:
                    log_error("Failed to checkout main from detached HEAD", e)

    def commit_all(self, message: str) -> str:
        self.repo.git.add(A=True)
        is_first_commit = not bool(self.repo.heads)
        
        if self.repo.is_dirty(untracked_files=True) or is_first_commit:
            commit = self.repo.index.commit(message)
            
            if is_first_commit:
                try:
                    if self.repo.active_branch.name != 'main':
                        self.repo.git.branch('-m', 'main')
                except Exception:
                    pass
                    
            return commit.hexsha
        return self.repo.head.commit.hexsha

    def ensure_dataset_branch(self, dataset_branch: str) -> None:
        """
        Checkout the branch used for this dataset (create from main if missing).
        No-op if there are no commits yet (first baseline will create main first).
        """
        if not self.repo.heads:
            return
        if dataset_branch == "main":
            self.repo.git.checkout("main")
            log_stage("Dataset path maps to branch 'main'; using main.")
            return
        self.repo.git.checkout("main")
        head_names = [h.name for h in self.repo.heads]
        if dataset_branch in head_names:
            self.repo.git.checkout(dataset_branch)
        else:
            self.repo.git.checkout("-b", dataset_branch)
        log_stage(f"Dataset work branch: {dataset_branch}")

    def ensure_dataset_branch_after_initial_commit(self, dataset_branch: str) -> None:
        """After the first-ever commit (created main), add/switch to the dataset branch."""
        if not self.repo.heads or dataset_branch == "main":
            return
        head_names = [h.name for h in self.repo.heads]
        if dataset_branch not in head_names:
            self.repo.git.branch(dataset_branch)
        self.repo.git.checkout(dataset_branch)
        log_stage(f"Switched to dataset branch: {dataset_branch}")

    def create_experiment_branch(self, iteration: int, base_branch: str):
        branch_name = f"experiment/iter_{iteration}"
        self.repo.git.checkout(base_branch)
        self.repo.git.checkout("-b", branch_name)
        return branch_name

    def checkout_branch(self, branch_name: str):
        self.repo.git.checkout(branch_name)

    def merge_to_dataset_branch(
        self, experiment_branch: str, dataset_branch: str, message: str
    ):
        self.repo.git.checkout(dataset_branch)
        self.repo.git.merge(experiment_branch, "--no-ff", "-m", message)
        return self.repo.head.commit.hexsha

    def discard_branch(self, experiment_branch: str, dataset_branch: str):
        self.repo.git.checkout(dataset_branch)
        self.repo.git.branch("-D", experiment_branch)

    def get_current_commit(self):
        return self.repo.head.commit.hexsha
