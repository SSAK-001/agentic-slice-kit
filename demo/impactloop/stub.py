from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel


# ---------------------------------------------------------
# Fake AI responses for the ImpactLoop MVP
# ---------------------------------------------------------

STUDENT_GOAL = """{
  "student_name": "Aarav",
  "goal": "I know basic Python and data analysis, but I do not have a real project that proves I can do it.",
  "current_skills": [
    "Python",
    "Excel",
    "Basic data analysis"
  ],
  "time_available": "6 hours per week",
  "desired_capability": "Data Analysis"
}"""


PROJECT_BRIEF = """{
  "project_title": "Analyse Student Cafeteria Spending Patterns",
  "problem_to_solve": "Understand how students spend money at the cafeteria and identify useful spending patterns.",
  "objective": "Use a small anonymised dataset to identify meaningful spending patterns and communicate the findings clearly.",
  "tasks": [
    "Inspect and clean the dataset",
    "Calculate basic spending statistics",
    "Identify important patterns",
    "Create two clear visualisations",
    "Write a short findings report"
  ],
  "deliverables": [
    "Cleaned dataset",
    "Analysis notebook",
    "Two visualisations",
    "Short findings report"
  ],
  "required_capabilities": [
    "Python",
    "Data analysis",
    "Data visualisation",
    "Communication"
  ],
  "evidence_requirements": [
    "Analysis notebook",
    "Charts",
    "Findings report"
  ]
}"""


# Deliberately weak evidence.
# The verifier should reject this.
WEAK_VERIFICATION = """{
  "status": "REVISION_REQUIRED",
  "reason": "The student provided only a written claim and did not provide enough evidence to demonstrate that the analysis was actually performed.",
  "missing_evidence": [
    "Analysis notebook",
    "Data visualisations",
    "Findings report"
  ]
}"""


# Stronger evidence.
# The verifier should accept this.
STRONG_VERIFICATION = """{
  "status": "PASS",
  "reason": "The submitted notebook, visualisations, and findings report provide evidence of the student's data-analysis contribution.",
  "missing_evidence": []
}"""


PROOF = """{
  "capability": "Data Analysis",
  "contribution": "Analysed anonymised student cafeteria spending data, identified spending patterns, and communicated the findings through visualisations and a short report.",
  "evidence": [
    "analysis_notebook.ipynb",
    "spending_patterns.png",
    "findings_report.md"
  ],
  "verification_status": "verified"
}"""


_SCRIPT: dict[str, list[str]] = {
    "intake": [STUDENT_GOAL],
    "decompose": [PROJECT_BRIEF],
    "verify": [WEAK_VERIFICATION, STRONG_VERIFICATION],
    "proof": [PROOF],
}


class Stub:
    """
    Fake replacement for slice.llm.complete.

    It gives us predictable AI responses so we can prove
    the workflow before using real API calls.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._n: dict[str, int] = {}

    def __call__(
        self,
        *,
        settings,
        budget,
        messages,
        schema: Type[BaseModel] | None = None,
        model: str | None = None,
        step: str = "call",
        timeout: float = 120.0,
    ) -> Any:

        base = step.split(":")[0]

        i = self._n.get(base, 0)
        self._n[base] = i + 1

        self.calls.append(step)

        try:
            raw = _SCRIPT[base][i]
        except (KeyError, IndexError):
            raise AssertionError(
                f"Stub has no scripted reply {i} for step {step!r}."
            )

        # Simulate token usage.
        budget.record_tokens(len(raw) // 4)

        # Validate the fake response using the real Pydantic schema.
        return schema.model_validate_json(raw) if schema else raw
