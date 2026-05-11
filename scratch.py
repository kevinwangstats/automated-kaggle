import git
import os
import shutil

if os.path.exists('test_repo'):
    shutil.rmtree('test_repo')
os.makedirs('test_repo')
repo = git.Repo.init('test_repo')
try:
    print(repo.active_branch.name)
except Exception as e:
    print("Error:", type(e), e)

