"""Central configuration — loads environment variables from .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
_repo_root = Path(__file__).resolve().parent.parent
load_dotenv(_repo_root / ".env")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# AWS
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Paths
DATA_DIR = _repo_root / "data"
OUTPUTS_DIR = _repo_root / "outputs"
