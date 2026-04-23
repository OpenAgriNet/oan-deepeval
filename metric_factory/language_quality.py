from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

from metric_factory.base import GEVAL_THRESHOLD, JUDGE_MODEL, LANGUAGE_LABELS


def language_quality_metric(lang_code: str, judge_model: str | None = None) -> GEval:
    lang_label = LANGUAGE_LABELS.get(lang_code, lang_code)
    model = judge_model or JUDGE_MODEL
    return GEval(
        name=f"LanguageQuality_{lang_code}",
        criteria=(
            f"User queried in {lang_label}. Evaluate across four dimensions:\n\n"

            f"A. LANGUAGE ADHERENCE (0–1): Response in {lang_label}?\n"
            f"   1.0 — Entire response in {lang_label}. Exempt: numbers, proper nouns, URLs, technical terms (PM Kisan, PMFBY, IMD, DAP, NPK).\n"
            f"   0.5 — Mostly {lang_label} but full sentences in another language.\n"
            f"   0.0 — Predominantly wrong language.\n\n"

            f"B. GRAMMAR (0–1): Correct {lang_label} grammar? Assess verb forms, postpositions, gender agreement. Penalise errors that impede understanding.\n\n"

            f"C. NATURALNESS (0–1): Natural, conversational {lang_label} a farmer would understand? "
            f"Penalise: machine-translated feel, literal English translations, awkward word order, inappropriate tone.\n\n"

            "D. STRUCTURE (0–1): Well-organised? Expected: answer → source citation (own line) → follow-up. Bullets only for eligibility lists. No repetition.\n\n"

            "Final = A×0.30 + B×0.25 + C×0.30 + D×0.15\n"
            "A = 0 VETO: final ≤ 0.15 regardless of other dimensions.\n\n"

            "## Expected Output Usage\n"
            "If provided: use as reference for ideal language, tone, structure. Don't require word-for-word match. "
            "If absent: evaluate on absolute criteria using INPUT language as reference."
        ),
        evaluation_steps=[
            f"Check EXPECTED OUTPUT: if provided, note language/tone/structure as benchmark. If absent, use INPUT and {lang_label} as reference. State path taken.",
            f"Confirm INPUT is {lang_label}. If not, flag but proceed with {lang_label} as expected response language.",
            f"Score A: What fraction is {lang_label}? Apply exemptions. Assign 1.0/0.5/0.0. Cross-check expected output if provided.",
            f"Score B: Evaluate {lang_label} grammar (verbs, postpositions, gender). Note errors. Compare to expected if provided. Assign B ∈ [0,1].",
            f"Score C: Naturalness check. Flag machine translation, calques, unnatural order. Compare phrasing to expected if provided. Assign C ∈ [0,1].",
            "Score D: Structure check (answer → citation line → follow-up). Bullets only for lists. Repetition check. Compare to expected if provided. Assign D ∈ [0,1].",
            "Apply veto: if A = 0, cap final at 0.15. Else: A×0.30 + B×0.25 + C×0.30 + D×0.15. Round to 2 decimals.",
            "State A, B, C, D, veto outcome, final. Cite one example per dimension. Note alignment/divergence from expected if used."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome=f"Wrong language (not {lang_label}), severe grammar, unintelligible. Veto applied: ≤0.15."),
            Rubric(score_range=(3, 5), expected_outcome=f"Mixed languages, grammar errors impede understanding, machine-translated feel, poor structure."),
            Rubric(score_range=(6, 8), expected_outcome=f"Mostly {lang_label} with minor slips, largely correct grammar, understandable but slightly unnatural, reasonable structure."),
            Rubric(score_range=(9, 10), expected_outcome=f"Entirely {lang_label} (exemptions ok), accurate grammar, natural farmer-friendly phrasing, proper structure (answer → citation line → follow-up)."),
        ],
        threshold=GEVAL_THRESHOLD,
        model=model,
        async_mode=True,
        verbose_mode=True,
    )