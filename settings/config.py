from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

BASE_URL: str = os.getenv("OAN_API_BASE_URL", "http://localhost:8000")
JUDGE_MODEL: str = os.getenv("DEEPEVAL_MODEL", "gpt-4o-mini")
MAX_WORKERS: int = int(os.getenv("OAN_EVAL_MAX_WORKERS", "500"))
GEVAL_THRESHOLD: float = float(os.getenv("OAN_EVAL_THRESHOLD", "0.6"))
PASS_RATE_THRESHOLD: float = float(os.getenv("OAN_PASS_RATE_THRESHOLD", "0.7"))

# Keep default dataset path stable so adding new test cases is easy.
DATASET_PATH: str = os.getenv(
    "OAN_EVAL_DATASET_PATH",
    "dataset/oan_eval_dataset.json",
)

MH_DATASET_PATH: str = os.getenv(
    "OAN_MH_EVAL_DATASET_PATH",
    "dataset/mh/oan_eval_dataset.json",
)
