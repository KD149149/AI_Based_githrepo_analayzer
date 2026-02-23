import os
import re
import shutil
import stat
from git import Repo


WORKSPACE_DIR = "workspace_repo"


# Windows safe delete
def _handle_remove_readonly(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clear_workspace():
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR, onerror=_handle_remove_readonly)


# Clone GitHub repo
def clone_repo(repo_url):
    clear_workspace()
    Repo.clone_from(repo_url, WORKSPACE_DIR)
    return WORKSPACE_DIR


# Clone local folder (copy)
def clone_local_folder(source_path):
    clear_workspace()
    shutil.copytree(source_path, WORKSPACE_DIR)
    return WORKSPACE_DIR


# Scan files
def scan_repository(repo_path):
    files_data = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".js", ".ts", ".env", ".txt")):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    files_data.append({
                        "file_path": file_path,
                        "content": content
                    })

                except Exception:
                    continue

    return files_data


# Secret detection
def find_hardcoded_secrets(files_data):
    patterns = [
        r"api[_-]?key\s*=\s*['\"].+['\"]",
        r"secret\s*=\s*['\"].+['\"]",
        r"password\s*=\s*['\"].+['\"]",
        r"token\s*=\s*['\"].+['\"]"
    ]

    detected = []

    for file in files_data:
        for pattern in patterns:
            if re.search(pattern, file["content"], re.IGNORECASE):
                detected.append({
                    "file": file["file_path"],
                    "pattern": pattern
                })

    return detected
