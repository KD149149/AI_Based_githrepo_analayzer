import os
import re
import shutil
from git import Repo


# -------- Clone Repository --------

def clone_repo(repo_url):
    target_dir = "cloned_repo"

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    Repo.clone_from(repo_url, target_dir)
    return target_dir


# -------- Scan Repository --------

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
                        "content_snippet": content[:1000]  # limit to first 1000 chars
                    })

                except Exception:
                    continue

    return files_data


# -------- Secret Detection --------

def find_hardcoded_secrets(repo_path):
    patterns = [
        r"api_key\s*=\s*['\"].+['\"]",
        r"secret\s*=\s*['\"].+['\"]",
        r"password\s*=\s*['\"].+['\"]",
        r"token\s*=\s*['\"].+['\"]"
    ]

    files_data = scan_repository(repo_path)
    detected = []

    for file in files_data:
        for pattern in patterns:
            if re.search(pattern, file["content_snippet"], re.IGNORECASE):
                detected.append({
                    "file": file["file_path"],
                    "pattern": pattern
                })

    return detected
