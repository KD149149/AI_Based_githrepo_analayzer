import os
import json
from config import OUTPUT_DIR


def generate_markdown(report_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/report.md", "w") as f:
        f.write("# Repository Engineering Analysis\n\n")
        f.write(f"Total Files: {report_data['total_files']}\n\n")
        f.write("## Architecture Summary\n")
        f.write(report_data["architecture"] + "\n\n")
        f.write("## Security Risks\n")
        f.write(report_data["security"] + "\n\n")
        f.write("## Performance Risks\n")
        f.write(report_data["performance"] + "\n\n")
        f.write("## Roadmap\n")
        f.write(report_data["roadmap"])


def generate_json(summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
