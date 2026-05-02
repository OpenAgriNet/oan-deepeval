"""
Integration tests for bharat-oan-api using DeepEval GEval metrics.

How to add new test cases quickly:
1. Open tests/evals/dataset/oan_eval_dataset.json
2. Add a new object with at least: name, input, target_lang/source_lang (or language)
3. Run: pytest tests/evals/test_integration.py -v --tb=short -s
"""

from __future__ import annotations

import pytest

from settings.config import BASE_URL, DATASET_PATH, PASS_RATE_THRESHOLD
from utils.dataset_loader import build_test_cases
from utils.evaluator import evaluate_case
from utils.execution import fetch_all_outputs
from models.models import OANTestCase
from utils.report import report_builder


_CASES: list[OANTestCase] | None = None
_OUTPUT_CACHE: dict[str, str | None] | None = None


def _get_cases() -> list[OANTestCase]:
    global _CASES
    if _CASES is None:
        _CASES = build_test_cases(DATASET_PATH)
    return _CASES


def _get_output_cache() -> dict[str, str | None]:
    global _OUTPUT_CACHE
    if _OUTPUT_CACHE is None:
        print("\n[Setup] Fetching API responses in parallel...")
        _OUTPUT_CACHE = fetch_all_outputs(_get_cases())
    return _OUTPUT_CACHE


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "tc" not in metafunc.fixturenames:
        return

    cases = _get_cases()
    cache = _get_output_cache()
    params = [(tc, cache.get(tc.name)) for tc in cases]
    ids = [tc.name for tc in cases]
    metafunc.parametrize("tc,actual_output", params, ids=ids)


@pytest.mark.integration
def test_oan_integration(tc: OANTestCase, actual_output: str | None) -> None:
    if not actual_output:
        report_builder.add_api_error(tc)
        assert actual_output, (
            f"[{tc.name}] API returned empty/None. "
            f"Check {BASE_URL}/api/health/live and session_id={tc.session_id}"
        )

    result = evaluate_case(tc, actual_output)
    report_builder.add_from_eval(tc, result, actual_output)

    assert result.is_ok, (
        f"[{tc.name}] Pass rate {result.pass_rate:.0%} "
        f"({result.passed}/{result.total} metrics passed, "
        f"threshold={int(PASS_RATE_THRESHOLD * 100)}%):\n"
        + "\n".join(result.failures)
    )

