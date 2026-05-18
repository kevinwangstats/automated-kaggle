import os
import datetime
import shutil

class WorkspaceManager:
    def __init__(self, dataset_branch: str, root_dir: str = ".workspaces"):
        self.root_dir = root_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workspace_name = f"{dataset_branch}_{timestamp}"
        self.workspace_dir = os.path.abspath(os.path.join(self.root_dir, self.workspace_name))
        
        os.makedirs(self.workspace_dir, exist_ok=True)
        print(f"[WorkspaceManager] Created isolated workspace at {self.workspace_dir}")

    def get_file_path(self, filename: str) -> str:
        """Returns the absolute path inside the workspace for a given filename."""
        return os.path.join(self.workspace_dir, filename)

    def read_file(self, filename: str) -> str:
        path = self.get_file_path(filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        return None

    def write_file(self, filename: str, content: str):
        path = self.get_file_path(filename)
        with open(path, "w") as f:
            f.write(content)

    def file_exists(self, filename: str) -> bool:
        return os.path.exists(self.get_file_path(filename))

    def cleanup(self):
        """Optional: cleans up the workspace"""
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)
