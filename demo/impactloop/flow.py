from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from slice.llm import complete
from slice.records import RunState

from .schema import (
    StudentGoal,
    ProjectBrief,
    VerificationResult,
    ProofOfAbility,
)

PROMPTS = Path(__file__).parent / "prompts"

MAX_REVISIONS = 1


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def build_flow(call=complete):

    def handle_drafting(ctx) -> RunState:

        # First pass: create student goal + project.
        if ctx.latest("student_goal") is None:

            student = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("intake"),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Create the StudentGoal record for this "
                            "demo student."
                        ),
                    },
                ],
                schema=StudentGoal,
                step="intake",
            )

            ctx.append(
                "student_goal",
                student.model_dump(),
                produced_by="agent:intake",
            )

            project = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("decompose"),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Create a practical micro-project for this "
                            "student goal:\n\n"
                            + student.model_dump_json(indent=2)
                        ),
                    },
                ],
                schema=ProjectBrief,
                step="decompose",
            )

            ctx.append(
                "project_brief",
                project.model_dump(),
                produced_by="agent:decomposer",
            )

            return RunState.GATING

        # Revision path.
        return RunState.GATING

    def handle_gating(ctx) -> RunState:

        project = ctx.latest("project_brief")

        result = call(
            settings=ctx.settings,
            budget=ctx.budget,
            messages=[
                {
                    "role": "system",
                    "content": load_prompt("verify"),
                },
                {
                    "role": "user",
                    "content": (
                        "Verify this project:\n\n"
                        + json.dumps(project, indent=2)
                    ),
                },
            ],
            schema=VerificationResult,
            step="verify",
        )

        ctx.append(
            "verification",
            result.model_dump(),
            produced_by="agent:verifier",
        )

        if result.status == "REVISION_REQUIRED":

            attempts = len(ctx.history("verification"))

            if attempts > MAX_REVISIONS:
                ctx.append(
                    "failure",
                    {
                        "kind": "verification_exhausted",
                        "detail": (
                            "The project did not pass verification "
                            "within the revision limit."
                        ),
                    },
                    produced_by="system",
                )

                return RunState.FAILED

            return RunState.DRAFTING

        # Verification passed, so create the Proof-of-Ability.
        proof = call(
            settings=ctx.settings,
            budget=ctx.budget,
            messages=[
                {
                    "role": "system",
                    "content": load_prompt("proof"),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "project": project,
                            "verification": result.model_dump(),
                        },
                        indent=2,
                    ),
                },
            ],
            schema=ProofOfAbility,
            step="proof",
        )

        ctx.append(
            "proof_of_ability",
            proof.model_dump(),
            produced_by="agent:proof",
        )

        return RunState.COMPLETE

    return SimpleNamespace(
        name="impactloop",
        handlers={
            RunState.DRAFTING: handle_drafting,
            RunState.GATING: handle_gating,
        },
    )
