"""
report.py — JSON report builder for OAN evaluation runs.

The PDF renderer lives in ``report_pdf.py`` and is re-exported here
for backward compatibility so existing imports keep working:

    from utils.report import save_report_pdf   # still works
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from settings.config import GEVAL_THRESHOLD, PASS_RATE_THRESHOLD
from utils.evaluator import CaseResult
from utils.report_pdf import save_report_pdf  # re-export
from models.models import LANGUAGE_LABELS, OANTestCase


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class ReportCase:
    name: str
    category: str
    section: str
    language: str
    language_label: str
    is_decline: bool
    question: str
    expected_output: str | None
    actual_output: str | None
    api_error: bool
    result: CaseResult | None
    metrics: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    status: str = "PASS"

    @property
    def pass_rate(self) -> float:
        return self.result.pass_rate if self.result else 0.0


# ── Report builder ─────────────────────────────────────────────────────────

class ReportBuilder:
    def __init__(self) -> None:
        self._cases: list[ReportCase] = []

    def add_api_error(self, tc: OANTestCase) -> None:
        case = ReportCase(
            name=tc.name,
            category=tc.category,
            section=tc.section,
            language=tc.language,
            language_label=LANGUAGE_LABELS.get(tc.language, tc.language),
            is_decline=tc.is_decline,
            question=tc.input,
            expected_output=tc.expected,
            actual_output=None,
            api_error=True,
            result=None,
            status="ERROR",
        )
        self._cases.append(case)

    def add_from_eval(
        self,
        tc: OANTestCase,
        result: CaseResult,
        actual_output: str,
    ) -> None:
        metrics = [
            {
                "name": mr.name,
                "score": mr.score,
                "threshold": mr.threshold,
                "status": mr.status,
                "reason": mr.reason,
            }
            for mr in result.metric_results
        ]
        case = ReportCase(
            name=tc.name,
            category=tc.category,
            section=tc.section,
            language=tc.language,
            language_label=LANGUAGE_LABELS.get(tc.language, tc.language),
            is_decline=tc.is_decline,
            question=tc.input,
            expected_output=tc.expected,
            actual_output=actual_output,
            api_error=False,
            result=result,
            metrics=metrics,
            failures=result.failures,
            status="PASS" if result.is_ok else "FAIL",
        )
        self._cases.append(case)

    def build(
        self,
        judge_model: str | None = None,
        run_meta: dict[str, Any] | None = None,
        meta_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._cases:
            return {"meta": {}, "summary": {}}

        total_cases = len(self._cases)
        passed_cases = sum(1 for c in self._cases if c.status == "PASS")
        failed_cases = sum(1 for c in self._cases if c.status == "FAIL")
        error_cases = sum(1 for c in self._cases if c.status == "ERROR")
        overall_pass_rate = passed_cases / total_cases if total_cases else 0.0

        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": judge_model or "gpt-4o-mini",
            "pass_rate_threshold": PASS_RATE_THRESHOLD,
            "geval_threshold": GEVAL_THRESHOLD,
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "error_cases": error_cases,
            "overall_pass_rate": overall_pass_rate,
            "overall_fail_rate": 1.0 - overall_pass_rate,
        }
        if run_meta:
            meta["run_meta"] = run_meta
        if meta_extra:
            meta.update(meta_extra)

        # ── Group by category ──────────────────────────────────────────
        by_category: dict[str, dict] = {}
        for case in self._cases:
            cat = case.category
            if cat not in by_category:
                by_category[cat] = {
                    "category": cat,
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "error": 0,
                    "pass_rate": 0.0,
                    "failure_rate": 0.0,
                    "failed_case_names": [],
                    "failure_summary": [],
                }
            bc = by_category[cat]
            bc["total"] += 1
            if case.status == "PASS":
                bc["passed"] += 1
            elif case.status == "FAIL":
                bc["failed"] += 1
                bc["failed_case_names"].append(case.name)
                for failure in case.failures:
                    bc["failure_summary"].append({
                        "metric": failure.split(" FAILED ")[0] if " FAILED " in failure else "Unknown",
                        "reason": failure,
                    })
            elif case.status == "ERROR":
                bc["error"] += 1

        for bc in by_category.values():
            bc["pass_rate"] = bc["passed"] / bc["total"] if bc["total"] else 0.0
            bc["failure_rate"] = 1.0 - bc["pass_rate"]

        # ── Group by section ──────────────────────────────────────────
        by_section: dict[str, dict] = {}
        for case in self._cases:
            sec = case.section or "(no section)"
            if sec not in by_section:
                by_section[sec] = {
                    "section": sec,
                    "total": 0, "passed": 0, "failed": 0, "error": 0,
                    "pass_rate": 0.0,
                }
            bs = by_section[sec]
            bs["total"] += 1
            if case.status == "PASS":
                bs["passed"] += 1
            elif case.status == "FAIL":
                bs["failed"] += 1
            elif case.status == "ERROR":
                bs["error"] += 1

        for bs in by_section.values():
            bs["pass_rate"] = bs["passed"] / bs["total"] if bs["total"] else 0.0

        # ── Group by metric ───────────────────────────────────────────
        by_metric: dict[str, dict] = {}
        for case in self._cases:
            for m in case.metrics:
                mname = m["name"]
                if mname not in by_metric:
                    by_metric[mname] = {
                        "metric": mname,
                        "total": 0, "passed": 0, "failed": 0,
                        "scores": [],
                        "avg_score": 0.0,
                        "pass_rate": 0.0,
                        "common_failure_reasons": [],
                    }
                bm = by_metric[mname]
                bm["total"] += 1
                bm["scores"].append(m.get("score", 0.0) or 0.0)
                if m["status"] == "PASS":
                    bm["passed"] += 1
                else:
                    bm["failed"] += 1
                    reason = m.get("reason", "")
                    if reason:
                        bm["common_failure_reasons"].append(reason)

        for bm in by_metric.values():
            bm["avg_score"] = sum(bm["scores"]) / len(bm["scores"]) if bm["scores"] else 0.0
            bm["pass_rate"] = bm["passed"] / bm["total"] if bm["total"] else 0.0
            del bm["scores"]  # don't persist raw scores list

        # ── Group by language ─────────────────────────────────────────
        by_language: dict[str, dict] = {}
        for case in self._cases:
            lang = case.language
            if lang not in by_language:
                by_language[lang] = {
                    "language": lang,
                    "language_label": case.language_label,
                    "total": 0, "passed": 0, "failed": 0, "error": 0,
                    "pass_rate": 0.0,
                }
            bl = by_language[lang]
            bl["total"] += 1
            if case.status == "PASS":
                bl["passed"] += 1
            elif case.status == "FAIL":
                bl["failed"] += 1
            elif case.status == "ERROR":
                bl["error"] += 1

        for bl in by_language.values():
            bl["pass_rate"] = bl["passed"] / bl["total"] if bl["total"] else 0.0

        # ── Top failing categories (sorted by failure rate desc) ──────
        top_failing = sorted(
            by_category.values(),
            key=lambda x: x.get("failure_rate", 0.0),
            reverse=True,
        )

        # ── Failed cases detail ───────────────────────────────────────
        failed_cases = [
            {
                "name": c.name,
                "category": c.category,
                "section": c.section,
                "language": c.language,
                "language_label": c.language_label,
                "is_decline": c.is_decline,
                "question": c.question,
                "expected_output": c.expected_output,
                "actual_output": c.actual_output or "",
                "pass_rate": c.pass_rate,
                "status": c.status,
                "api_error": c.api_error,
                "metrics": c.metrics,
                "failures": c.failures,
            }
            for c in self._cases if c.status in ("FAIL", "ERROR")
        ]

        summary = {
            "by_category": list(by_category.values()),
            "by_section": list(by_section.values()),
            "by_metric": list(by_metric.values()),
            "by_language": list(by_language.values()),
            "top_failing_categories": top_failing,
        }

        return {"meta": meta, "summary": summary, "failed_cases": failed_cases}


# Global instance
report_builder = ReportBuilder()


# ── JSON persistence ───────────────────────────────────────────────────────

def save_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Save the report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path