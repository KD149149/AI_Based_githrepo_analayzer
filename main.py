

import argparse
import os
import json

from tools import clone_repo, scan_repository, find_hardcoded_secrets


def main():
    parser = argparse.ArgumentParser(description="AI Repo Analyzer")

    parser.add_argument(
        "--repo",
        type=str,
        help="GitHub repository URL"
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Local project folder path"
    )

    args = parser.parse_args()

    # Validate input
    if not args.repo and not args.path:
        print("Error: Please provide either --repo or --path")
        return

    # Determine source
    if args.repo:
        print("Using remote repository...")
        repo_path = clone_repo(args.repo)
    else:
        print("Using local project path...")
        if not os.path.exists(args.path):
            print("Error: Provided path does not exist.")
            return
        repo_path = args.path

    # Run analysis
    print("Scanning repository...")
    files_data = scan_repository(repo_path)

    print("Detecting hardcoded secrets...")
    secrets = find_hardcoded_secrets(repo_path)

    # Ensure output folder exists
    os.makedirs("output", exist_ok=True)

    # Generate report.md
    report_path = os.path.join("output", "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Repository Analysis Report\n\n")
        f.write(f"Total Files Scanned: {len(files_data)}\n\n")
        f.write(f"Potential Secrets Found: {len(secrets)}\n\n")

        if secrets:
            f.write("## Detected Secrets\n")
            for secret in secrets:
                f.write(f"- {secret}\n")

    # Generate summary.json
    summary = {
        "total_files": len(files_data),
        "secrets_found": len(secrets),
        "secret_details": secrets
    }

    summary_path = os.path.join("output", "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Analysis complete. See output/report.md and output/summary.json")


if __name__ == "__main__":
    main()
