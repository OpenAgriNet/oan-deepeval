from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric
from metric_factory.base import GEVAL_THRESHOLD, JUDGE_MODEL


def response_quality_metric(judge_model: str | None = None) -> GEval:
    model = judge_model or JUDGE_MODEL
    return GEval(
        name="ResponseQuality",
        criteria=(
            "Evaluate agricultural assistant response quality across completeness and citation.\n\n"

            "## Classification\n"
            "FULL — Substantive answer with key facts | PARTIAL — Answered but gaps exist | "
            "DEFLECT — Only redirect/clarification | EMPTY — No answer | "
            "NO_DATA — Correctly states data unavailable (valid response)\n\n"

            "## Negative Case Detection\n"
            "NEGATIVE CASE = EXPECTED OUTPUT or ACTUAL OUTPUT signals data unavailability "
            "('not found', 'unavailable', 'no records', etc. in any language). "
            "LLM should acknowledge unavailability, NOT fabricate data.\n\n"

            "## Dimension A: Completeness (0–1)\n"
            "NEGATIVE CASE anchors:\n"
            "  1.0 — Clearly states unavailability + reason/alternative\n"
            "  0.7 — Clearly states unavailability, no further guidance\n"
            "  0.4 — Vaguely implies missing data\n"
            "  0.0 — FABRICATES data (critical failure)\n\n"
            "POSITIVE CASE checklists:\n"
            "  Mandi: commodity + price + unit + market + recency\n"
            "  Weather: location + temp(unit) + conditions + advisory\n"
            "  Scheme: name + eligibility/status + next step\n"
            "  Pest/crop: problem + treatment + dose/method + timing\n"
            "Scores: 1.0 all present | 0.7 one minor gap | 0.4 major gap | 0.0 DEFLECT/EMPTY\n"
            "Deduct 0.1 if ≥3 unrelated/repetitive sentences.\n\n"

            "## Dimension B: Citation (0–1)\n"
            "  1.0 — Bold citation on own line (any language)\n"
            "  0.5 — Mentioned but not bolded OR not on own line\n"
            "  0.0 — No source\n"
            "Mark B = N/A (exclude from score) if: DEFLECT/EMPTY, NEGATIVE CASE with no source consulted, or purely procedural query.\n\n"

            "## Final Score\n"
            "B scored → A×0.65 + B×0.35 | B = N/A → A×1.00\n\n"

            "## Expected Output Usage\n"
            "If provided: check for negative case signals first, then use as reference for key facts/checklist. "
            "If absent: score against query-type checklist; infer negative case from actual output if present."
        ),
        evaluation_steps=[
            "Check EXPECTED OUTPUT: if provided, detect negative case signals ('not found', etc.) and list key facts. If absent, state 'evaluating on absolute criteria' and check ACTUAL OUTPUT for unavailability language.",
            "Classify ACTUAL OUTPUT as FULL/PARTIAL/DEFLECT/EMPTY/NO_DATA with reason. Remember NO_DATA is valid.",
            "Identify query type and checklist. For NEGATIVE CASE: note expected behavior is acknowledging unavailability. For POSITIVE: list required items and mark each present/missing/partial.",
            "Score A: NEGATIVE CASE uses {1.0, 0.7, 0.4, 0.0} anchors. POSITIVE CASE uses checklist + padding rule. State score and evidence.",
            "Score B: Check exemption. If not exempt, locate citation (bold? own line?) and assign {0.0, 0.5, 1.0}. Quote evidence.",
            "Compute final: B scored → A×0.65 + B×0.35, B = N/A → A×1.00. State A, B, final.",
            "Verdict (2 sentences): classification, score, driver. For NEGATIVE: did LLM handle unavailability correctly? For POSITIVE: what prevented completeness? Note comparison to expected output if used."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        rubric=[
            Rubric(score_range=(0, 1), expected_outcome="DEFLECT/EMPTY or NEGATIVE CASE fabrication. Score ≤0.10."),
            Rubric(score_range=(2, 4), expected_outcome="PARTIAL with major gap or no citation when required. NEGATIVE CASE: vague acknowledgment, no guidance."),
            Rubric(score_range=(5, 7), expected_outcome="PARTIAL with minor gap or citation formatting issue. NEGATIVE CASE: unavailability stated, no alternative."),
            Rubric(score_range=(8, 10), expected_outcome="FULL with all items + proper citation. NEGATIVE CASE: clear acknowledgment + reason/alternative."),
        ],
        threshold=GEVAL_THRESHOLD,
        model=model,
        async_mode=True,
        verbose_mode=True,
    )