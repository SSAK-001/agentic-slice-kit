from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from slice.llm import complete
from slice.records import RunState

from .schema import (
    StudentGoal,
    ProjectBrief,
    StudentMatch,
    TeamProposal,
    TaskPlan,
    MentorDecision,
    VerificationResult,
    ProofOfAbility,
    OpportunityRecommendation,
)

PROMPTS = Path(__file__).parent / "prompts"

MAX_REVISIONS = 1


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def build_flow(call=complete):

    def handle_drafting(ctx) -> RunState:
        # Step 1: Intake agent
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
                            "Create the StudentGoal record for the "
                            "Student Event Discovery challenge."
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

            # Step 2: Problem Decomposer agent
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
                            "Create a project brief for this challenge:\n\n"
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

            # Step 3: Semantic Team Matcher agent
            team = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Match students to complementary roles. "
                            "Explain why each person fits."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            project.model_dump(),
                            indent=2,
                        ),
                    },
                ],
                schema=TeamProposal,
                step="team",
            )

            ctx.append(
                "team_proposal",
                team.model_dump(),
                produced_by="agent:team_matcher",
            )

            # Step 4: Orchestrator agent
            plan = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Create a project plan with task owners, "
                            "acceptance conditions, and evidence requirements."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "project": project.model_dump(),
                                "team": team.model_dump(),
                            },
                            indent=2,
                        ),
                    },
                ],
                schema=TaskPlan,
                step="plan",
            )

            ctx.append(
                "task_plan",
                plan.model_dump(),
                produced_by="agent:orchestrator",
            )

            return RunState.PROBING

        # After evidence revision, return to the mentor/planning stage.
        return RunState.PROBING

    def handle_probing(ctx) -> RunState:
        # Step 5: Mentor/Blocker Handler agent
        if ctx.latest("mentor_decision") is None:
            decision = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Identify the important project priority "
                            "and record the mentor decision."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "project": ctx.latest("project_brief"),
                                "team": ctx.latest("team_proposal"),
                                "plan": ctx.latest("task_plan"),
                            },
                            indent=2,
                        ),
                    },
                ],
                schema=MentorDecision,
                step="mentor",
            )

            ctx.append(
                "mentor_decision",
                decision.model_dump(),
                produced_by="agent:mentor_handler",
            )

        return RunState.GATING

    def handle_gating(ctx) -> RunState:
        # Step 6: Verifier agent
        project = ctx.latest("project_brief")
        plan = ctx.latest("task_plan")
        mentor = ctx.latest("mentor_decision")

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
                    "content": json.dumps(
                        {
                            "project": project,
                            "task_plan": plan,
                            "mentor_decision": mentor,
                        },
                        indent=2,
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

        # Step 7: Proof-of-Ability agent
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

        # Step 8: Connector agent
        recommendation = call(
            settings=ctx.settings,
            budget=ctx.budget,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Recommend a suitable next opportunity based "
                        "only on the verified proof."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        proof.model_dump(),
                        indent=2,
                    ),
                },
            ],
            schema=OpportunityRecommendation,
            step="connector",
        )

        ctx.append(
            "opportunity_recommendation",
            recommendation.model_dump(),
            produced_by="agent:connector",
        )

        return RunState.COMPLETE

    return SimpleNamespace(
        name="impactloop",
        handlers={
            RunState.DRAFTING: handle_drafting,
            RunState.PROBING: handle_probing,
            RunState.GATING: handle_gating,
        },
    )
