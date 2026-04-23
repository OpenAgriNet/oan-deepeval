from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from deepeval.test_case import LLMTestCase

from settings.config import JUDGE_MODEL
from metric_factory import get_metrics_for_case
from models.models import LANGUAGE_LABELS, OANTestCase

PASS_RATE_THRESHOLD: float = 0.70  # case passes if >= 70 % of metrics pass


@dataclass
class MetricResult:
    name: str
    score: float
    threshold: float
    status: str           # "PASS" | "FAIL"
    reason: str
    rubric_range: str     # e.g. "9-10"
    rubric_description: str


def _find_rubric(metric, raw_score: int) -> tuple[str, str]:
    """Return (range_str, expected_outcome) for the matching rubric bucket."""
    rubrics = getattr(metric, "rubric", None) or getattr(metric, "rubrics", None)
    if not rubrics:
        return ("", "")
    for r in rubrics:
        lo, hi = r.score_range
        if lo <= raw_score <= hi:
            return (f"{lo}-{hi}", r.expected_outcome)
    return ("", "")


@dataclass
class CaseResult:
    passed: int
    total: int
    failures: list[str]
    metric_results: list[MetricResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def is_ok(self) -> bool:
        return self.pass_rate >= PASS_RATE_THRESHOLD


def evaluate_case(
    tc: OANTestCase,
    actual_output: str,
    *,
    judge_model: str | None = None,
) -> CaseResult:
    """Run all applicable metrics on a single test case via deepeval.evaluate().

    Returns a CaseResult; the test passes when pass_rate >= PASS_RATE_THRESHOLD.
    """
    case = LLMTestCase(
        input=tc.input,
        actual_output=actual_output,
        expected_output=tc.expected,
    )
    metrics = get_metrics_for_case(
        is_decline=tc.is_decline,
        lang_code=tc.language,
        judge_model=judge_model,
    )

    async def _measure_all() -> None:
        await asyncio.gather(*[m.a_measure(case) for m in metrics])

    asyncio.run(_measure_all())

    lang_label = LANGUAGE_LABELS.get(tc.language, tc.language)
    path = "decline" if tc.is_decline else f"category={tc.category}"

    results_log: list[str] = []
    failures: list[str] = []
    metric_results: list[MetricResult] = []
    passed = 0

    for metric in metrics:
        ok = metric.is_successful()
        status = "PASS" if ok else "FAIL"
        raw_score = round((metric.score or 0.0) * 10)
        rubric_range, rubric_desc = _find_rubric(metric, raw_score)
        results_log.append(f"  {metric.name:<35} {(metric.score or 0.0):.3f} [{status}] {metric.reason}")
        metric_results.append(
            MetricResult(
                name=metric.name,
                score=metric.score or 0.0,
                threshold=getattr(metric, "threshold", PASS_RATE_THRESHOLD),
                status=status,
                reason=metric.reason or "",
                rubric_range=rubric_range,
                rubric_description=rubric_desc,
            )
        )
        if ok:
            passed += 1
        else:
            failures.append(
                f"{metric.name} FAILED (score={(metric.score or 0.0):.3f}): {metric.reason}"
            )

    result = CaseResult(
        passed=passed,
        total=len(metrics),
        failures=failures,
        metric_results=metric_results,
    )
    print(
        f"\n[{tc.name}] ({path}) lang={lang_label}\n"
        f"  session_id: {tc.session_id}\n"
        f"  Judge model: {judge_model or JUDGE_MODEL}\n"
        f"  Pass rate: {result.passed}/{result.total} ({result.pass_rate:.0%})"
        f" [{'OK' if result.is_ok else 'FAIL'}]\n"
        + "\n".join(results_log)
    )
    return result
