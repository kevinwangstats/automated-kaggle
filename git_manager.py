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
        try:
            if self.repo.is_dirty(untracked_files=True):
                commit = self.repo.index.commit(message)
                return commit.hexsha
            # If not dirty, but no commits yet (initial repo)
            if not self.repo.heads:
                commit = self.repo.index.commit(message)
                try:
                    if self.repo.active_branch.name != 'main':
                        self.repo.git.branch('-m', 'main')
                except Exception:
                    pass
                return commit.hexsha
        except Exception as e:
            log_error("Commit failed", e)
            
        return self.get_current_commit() if self.repo.heads else ""

    def revert_changes(self):
        """Discards all local changes in the working directory."""
        try:
            self.repo.git.checkout(".")
            self.repo.git.clean("-fd")
            log_stage("Reverted local changes and cleaned working directory.")
        except Exception as e:
            log_error("Failed to revert changes", e)

    def discard_branch(self, experiment_branch: str, dataset_branch: str):
        """Switches back to dataset_branch and deletes the experiment_branch, discarding changes."""
        try:
            self.repo.git.checkout(dataset_branch, force=True)
            self.repo.git.branch("-D", experiment_branch)
            log_stage(f"Discarded experiment branch {experiment_branch}")
        except Exception as e:
            log_error(f"Failed to discard branch {experiment_branch}", e)
            # Fallback attempt to just get back to a safe state
            self.checkout_branch(dataset_branch)

    def get_current_commit(self):
        return self.repo.head.commit.hexsha
