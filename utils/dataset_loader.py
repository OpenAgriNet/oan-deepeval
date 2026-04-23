from __future__ import annotations

import json
import re
from pathlib import Path

from models.models import OANTestCase


def _normalize_item(item: dict) -> OANTestCase:
    language = item.get("language") or item.get("target_lang") or item.get("source_lang") or "en"
    question = item.get("question") or item.get("input")
    expected = item.get("expected_output") or item.get("expected_answer") or None
    if not question:
        raise ValueError(f"Dataset item {item.get('name', '<unknown>')} is missing question/input")

    name = item.get("name")
    if not name:
        category = item.get("category", "general")
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
        slug = slug[:40] if slug else "query"
        name = f"{category}_{language}_{slug}"

    return OANTestCase(
        name=name,
        category=item.get("category", "general"),
        language=language,
        input=question,
        expected=expected,
        is_decline=item.get("is_decline", False),
        context=item.get("context", []),
    )


def _clean_section(raw: str) -> str:
    """Strip leading/trailing '===' markers and whitespace."""
    return re.sub(r"(^[=\s]+)|([=\s]+$)", "", raw).strip()


def build_test_cases(dataset_path: str) -> list[OANTestCase]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of test cases")

    current_section: str = ""
    cases: list[OANTestCase] = []
    for item in data:
        if "_section" in item:
            current_section = _clean_section(item["_section"])
            continue
        tc = _normalize_item(item)
        tc.section = current_section
        cases.append(tc)

    print(f"[Dataset] Loaded {len(cases)} test cases from {path}")
    return cases
