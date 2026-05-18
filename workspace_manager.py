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

    def get_previous_workspace(self) -> str:
        """Finds the most recent workspace directory for the same dataset branch, excluding the current one."""
        if not os.path.exists(self.root_dir):
            return None
        prefix = self.workspace_name.rsplit('_', 2)[0] # get dataset_branch part
        dirs = [d for d in os.listdir(self.root_dir) if d.startswith(f"{prefix}_")]
        dirs.sort()
        if self.workspace_name in dirs:
            idx = dirs.index(self.workspace_name)
            if idx > 0:
                return os.path.join(self.root_dir, dirs[idx - 1])
        return None

    def copy_from_previous(self, previous_dir: str, filenames: list):
        """Copies specified files from a previous workspace to the current one."""
        for fname in filenames:
            src = os.path.join(previous_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, self.get_file_path(fname))
