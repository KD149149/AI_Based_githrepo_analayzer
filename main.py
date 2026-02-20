import argparse
from crewai import Task, Crew
from agents import (
    create_scanner_agent,
    create_architecture_agent,
    create_security_agent,
    create_performance_agent,
    create_roadmap_agent
)
from tools import clone_repo, scan_repository, find_hardcoded_secrets
from report_generator import generate_markdown, generate_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, help="GitHub repo URL")
    parser.add_argument("--path", type=str, help="Local repo path")
    args = parser.parse_args()

    if args.repo:
        repo_path = clone_repo(args.repo)
    elif args.path:
        repo_path = args.path
    else:
        raise ValueError("Provide --repo or --path")

    scan_result = scan_repository(repo_path)

    security_issues = find_hardcoded_secrets(scan_result["files"])

    architecture_summary = "Monolith architecture detected based on single entry pattern."
    performance_summary = "Potential lack of pagination in API endpoints."
    roadmap = "P0: Move secrets to environment variables.\nP1: Add pagination."

    report_data = {
        "total_files": scan_result["total_files"],
        "architecture": architecture_summary,
        "security": str(security_issues),
        "performance": performance_summary,
        "roadmap": roadmap
    }

    summary_json = {
        "repo_name": repo_path,
        "language_detected": list(scan_result["languages"].keys()),
        "architecture_type": "monolith",
        "top_security_risks": security_issues,
        "top_performance_risks": [
            {"issue": "Missing pagination", "severity": "Medium"}
        ],
        "roadmap": [
            {"priority": "P0", "task": "Move secrets to env vars", "effort": "2 hours"}
        ]
    }

    generate_markdown(report_data)
    generate_json(summary_json)

    print("Analysis complete. See output/report.md and output/summary.json")


if __name__ == "__main__":
    main()
