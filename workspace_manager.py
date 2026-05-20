"""
workspace_manager.py

Manages isolated execution environments for the pipeline.
Designed as a library module to handle ephemeral file paths safely.
Usage:
    >>> from workspace_manager import WorkspaceManager
    >>> wm = WorkspaceManager("dataset_branch")
"""
import os
import shutil
from pathlib import Path

class WorkspaceManager:
    def __init__(self, dataset_branch: str, root_dir: str = ".workspaces"):
        self.root_dir = Path(root_dir)
        self.workspace_name = dataset_branch
        self.workspace_dir = (self.root_dir / self.workspace_name).resolve()
        
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        print(f"[WorkspaceManager] Workspace at {self.workspace_dir}")

    def get_file_path(self, filename: str) -> str:
        """Returns the absolute path inside the workspace for a given filename."""
        return str(self.workspace_dir / filename)

    def read_file(self, filename: str) -> str:
        path = self.workspace_dir / filename
        if path.exists():
            with open(path, "r") as f:
                return f.read()
        return None

    def write_file(self, filename: str, content: str):
        path = self.workspace_dir / filename
        with open(path, "w") as f:
            f.write(content)

    def file_exists(self, filename: str) -> bool:
        return (self.workspace_dir / filename).exists()

    def cleanup(self):
        """Optional: cleans up the workspace"""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
