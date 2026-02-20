import os
MODEL_PROVIDER = "local"
MODEL_NAME = "llama3"

# LLM Configuration
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")  # openai | azure | local
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Repo scanning config
MAX_FILE_SIZE_MB = 1

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "venv",
    ".venv",
    "coverage",
    "target"
}

OUTPUT_DIR = "output"
