import os
import re
import json
import subprocess
from config import IGNORE_DIRS, MAX_FILE_SIZE_MB


from git import Repo
import os
import shutil

def clone_repo(repo_url):
    target_dir = "cloned_repo"

    if os.path.exists(target_dir):
        print("Deleting old repo and cloning fresh...")
        shutil.rmtree(target_dir)

    print("Cloning repository...")
    Repo.clone_from(repo_url, target_dir)

    return target_dir




def is_valid_file(file_path):
    if any(part in IGNORE_DIRS for part in file_path.split(os.sep)):
        return False
    if os.path.getsize(file_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False
    return True


def scan_repository(path):
    file_data = []
    languages = {}

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            full_path = os.path.join(root, file)
            if not is_valid_file(full_path):
                continue

            ext = file.split(".")[-1] if "." in file else "unknown"
            languages[ext] = languages.get(ext, 0) + 1

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except:
                content = ""

            file_data.append({
                "path": full_path,
                "extension": ext,
                "content_snippet": content[:2000]
            })

    return {
        "total_files": len(file_data),
        "languages": languages,
        "files": file_data
    }


def find_hardcoded_secrets(files):
    patterns = [
        r'api_key\s*=\s*["\'].*["\']',
        r'password\s*=\s*["\'].*["\']',
        r'JWT_SECRET\s*=\s*["\'].*["\']'
    ]

    issues = []

    for file in files:
        for pattern in patterns:
            if re.search(pattern, file["content_snippet"], re.IGNORECASE):
                issues.append({
                    "issue": "Hardcoded secret detected",
                    "file": file["path"],
                    "severity": "High"
                })

    return issues
