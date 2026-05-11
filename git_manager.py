import git
import os
from logger import log_stage, log_error

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
        if self.repo.heads:
            try:
                if self.repo.active_branch.name != 'main':
                    self.repo.git.checkout('-B', 'main')
            except Exception as e:
                log_error("Failed to checkout main branch", e)

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

    def create_experiment_branch(self, iteration: int):
        branch_name = f"experiment/iter_{iteration}"
        # Always branch from main
        self.repo.git.checkout('main')
        self.repo.git.checkout('-b', branch_name)
        return branch_name

    def checkout_branch(self, branch_name: str):
        self.repo.git.checkout(branch_name)

    def merge_to_main(self, branch_name: str, message: str):
        self.repo.git.checkout('main')
        self.repo.git.merge(branch_name, '--no-ff', '-m', message)
        return self.repo.head.commit.hexsha

    def discard_branch(self, branch_name: str):
        self.repo.git.checkout('main')
        self.repo.git.branch('-D', branch_name)

    def get_current_commit(self):
        return self.repo.head.commit.hexsha
