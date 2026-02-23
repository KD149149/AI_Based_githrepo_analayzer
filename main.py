import argparse
import os
import json
from tools import clone_repo, clone_local_folder, scan_repository, find_hardcoded_secrets


def main():
    parser = argparse.ArgumentParser(description="AI Repo Analyzer")

    parser.add_argument("--repo", type=str, help="GitHub repository URL")
    parser.add_argument("--path", type=str, help="Local project folder path")

    args = parser.parse_args()

    if not args.repo and not args.path:
        print("Error: Please provide either --repo or --path")
        return

    # Remote repository mode
    if args.repo:
        print("Cloning remote repository...")
        repo_path = clone_repo(args.repo)

    # Local project mode (clone locally)
    else:
        local_path = os.path.abspath(args.path)

        if not os.path.exists(local_path):
            print(f"Error: Provided path does not exist -> {local_path}")
            return

        print("Cloning local folder into workspace...")
        repo_path = clone_local_folder(local_path)

    # Run analysis
    print("Scanning repository...")
    files_data = scan_repository(repo_path)

    print("Detecting hardcoded secrets...")
    secrets = find_hardcoded_secrets(files_data)

    # Output directory
    os.makedirs("output", exist_ok=True)

    with open("output/report.md", "w", encoding="utf-8") as f:
        f.write("# Repository Analysis Report\n\n")
        f.write(f"Total Files Scanned: {len(files_data)}\n\n")
        f.write(f"Secrets Found: {len(secrets)}\n\n")

        if secrets:
            f.write("## Secret Details\n")
            for s in secrets:
                f.write(f"- File: {s['file']}\n")

    summary = {
        "total_files": len(files_data),
        "secrets_found": len(secrets),
        "secret_details": secrets
    }

    with open("output/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Analysis complete. See output/report.md and output/summary.json")


if __name__ == "__main__":
    main()
