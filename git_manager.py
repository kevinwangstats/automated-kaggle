"""
git_manager.py

Handles isolated Git operations for data-branch versioning.
Designed as an internal library module to provide the GitManager class.
Usage:
    >>> from git_manager import GitManager
    >>> gm = GitManager()
"""
import git
import os
import re
from pathlib import Path

from logger import log_stage, log_error, log_info


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
        self.repo_path = Path(repo_path).resolve()
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

    def ensure_dataset_branch(self, dataset_branch: str) -> None:
        """
        Checkout the branch used for this dataset (create from main if missing).
        No-op if there are no commits yet (first baseline will create main first).
        """
        if not self.repo.heads:
            return
            
        try:
            if self.repo.active_branch.name == dataset_branch:
                log_stage(f"Already on dataset work branch: {dataset_branch}")
                return
        except TypeError:
            pass # Detached HEAD, continue with normal logic

        if dataset_branch == "main":
            try:
                self.repo.git.checkout("main")
            except git.exc.GitCommandError as e:
                if "already checked out" in str(e).lower() or "worktree" in str(e).lower():
                    log_error(f"The branch 'main' is already checked out in another directory (git worktree).")
                    raise
                else:
                    raise
            log_stage("Dataset path maps to branch 'main'; using main.")
            return
            
        head_names = [h.name for h in self.repo.heads]
        if dataset_branch in head_names:
            try:
                self.repo.git.checkout(dataset_branch)
            except git.exc.GitCommandError as e:
                if "already checked out" in str(e).lower() or "worktree" in str(e).lower():
                    log_error(f"[CRITICAL ERROR] The dataset branch '{dataset_branch}' is already checked out in another directory (git worktree).")
                    log_info("Please navigate to your active worktree directory to run this model safely without conflict.")
                    raise
                else:
                    raise
        else:
            self.repo.git.checkout("-b", dataset_branch, "main")
            
        log_stage(f"Dataset work branch: {dataset_branch}")

    def ensure_dataset_branch_after_initial_commit(self, dataset_branch: str) -> None:
        """After the first-ever commit (created main), add/switch to the dataset branch."""
        if not self.repo.heads or dataset_branch == "main":
            return
        head_names = [h.name for h in self.repo.heads]
        if dataset_branch not in head_names:
            self.repo.git.branch(dataset_branch)
        try:
            self.repo.git.checkout(dataset_branch)
        except git.exc.GitCommandError as e:
            if "already checked out" in str(e).lower() or "worktree" in str(e).lower():
                log_error(f"The branch '{dataset_branch}' is already checked out in another directory.")
                raise
            else:
                raise
        log_stage(f"Switched to dataset branch: {dataset_branch}")

    def checkout_branch(self, branch_name: str):
        try:
            self.repo.git.checkout(branch_name)
        except git.exc.GitCommandError as e:
            if "already checked out" in str(e).lower() or "worktree" in str(e).lower():
                log_error(f"[CRITICAL ERROR] The branch '{branch_name}' is already checked out in another directory (git worktree).")
                log_info("Please navigate to your active worktree directory to run this model safely without conflict.")
                raise
            else:
                raise

    def revert_changes(self):
        """Discards all local changes in the working directory."""
        try:
            self.repo.git.checkout(".")
            self.repo.git.clean("-fd")
            log_stage("Reverted local changes and cleaned working directory.")
        except Exception as e:
            log_error("Failed to revert changes", e)

    def get_current_commit(self):
        return self.repo.head.commit.hexsha

    def is_on_main(self) -> bool:
        if not self.repo.heads:
            return False
        try:
            return self.repo.active_branch.name == 'main'
        except TypeError:
            return False # Detached HEAD

    def has_uncommitted_changes(self) -> bool:
        return self.repo.is_dirty(untracked_files=True)

    def branch_exists(self, branch_name: str) -> bool:
        if not self.repo.heads:
            return False
        return branch_name in [h.name for h in self.repo.heads]

    def is_branch_based_on_latest_main(self, branch_name: str) -> bool:
        if not self.branch_exists(branch_name) or not self.branch_exists('main'):
            return True
        try:
            # Check if main is an ancestor of the branch
            return self.repo.is_ancestor(self.repo.heads.main.commit, self.repo.heads[branch_name].commit)
        except Exception:
            return False
            
    def delete_branch(self, branch_name: str):
        try:
            self.repo.git.branch('-D', branch_name)
        except Exception as e:
            log_error(f"Failed to delete branch {branch_name}", e)

    def merge_main(self):
        try:
            log_stage("Merging latest 'main' into active branch...")
            self.repo.git.merge('main')
            log_stage("Successfully merged 'main'.")
        except git.exc.GitCommandError as e:
            log_error("Auto-merge of 'main' failed due to git conflicts. Please resolve manually or start fresh.", e)
            raise
