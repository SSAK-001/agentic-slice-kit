from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from slice import callback
from slice.llm import complete
from slice.records import RunState

from .schema import (
    StudentGoal,
    ProjectBrief,
    TeamProposal,
    TaskPlan,
    VerificationResult,
    ProofOfAbility,
    OpportunityRecommendation,
)


PROMPTS = Path(__file__).parent / "prompts"

# This is a domain revision limit, not an API-credit limit.
MAX_REVISIONS = 1


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def build_flow(call=complete):

    def handle_drafting(ctx) -> RunState:
        """
        Build the student goal, project brief, team proposal,
        and task plan.
        """

        # ---------------------------------------------------------
        # Agent 1: Intake
        # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # Agent 2: Problem Decomposer
            # ---------------------------------------------------------
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
                            "Create a project brief for this "
                            "Student Event Discovery challenge:\n\n"
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

            # ---------------------------------------------------------
            # Agent 3: Semantic Team Matcher
            # ---------------------------------------------------------
            team = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Match students to complementary roles. "
                            "Explain why each student fits the project."
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

            # ---------------------------------------------------------
            # Agent 4: Orchestrator
            # ---------------------------------------------------------
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

        # After weak evidence, return to the probing stage.
        return RunState.PROBING

    def handle_probing(ctx) -> RunState:
        """
        Human-in-the-loop stage.

        The first time this stage runs, it creates a real persisted
        mentor question and suspends the workflow.

        After the mentor answers, the callback system wakes the run.
        The answer is then converted into a structured mentor_decision
        record that later agents can read.
        """

        # If a structured mentor decision already exists, do not create
        # another one after an evidence revision.
        if ctx.latest("mentor_decision") is not None:
            return RunState.GATING

        # Check whether a human has already answered the persisted question.
        answers = ctx.history("expert_answer")

        if not answers:
            callback.ask(
                ctx.store,
                ctx.run_id,
                (
                    "Which priority should guide the project if a unified "
                    "discovery flow conflicts with existing club-channel "
                    "preferences?"
                ),
                {
                    "resume_state": RunState.PROBING.value,
                    "options": [
                        "Prioritise one unified student experience",
                        "Keep every existing channel unchanged",
                    ],
                    "reason": (
                        "The system detected a possible conflict between "
                        "a unified discovery experience and existing club channels."
                    ),
                },
                ctx.settings,
            )

            # The runner will stop here with AWAITING_EXPERT.
            return RunState.AWAITING_EXPERT

        # Convert the human's persisted answer into a typed project record.
        answer = answers[-1].payload

        ctx.append(
            "mentor_decision",
            {
                "question": answer["question"],
                "decision": answer.get("answer") or "No mentor response",
                "priority": answer.get("answer") or "Unresolved",
                "answered_by": answer.get("who") or "unresolved_no_expert",
            },
            produced_by="human:mentor",
        )

        return RunState.GATING

    def handle_gating(ctx) -> RunState:
        """
        Verify evidence, request revision if necessary,
        then create Proof-of-Ability and recommend an opportunity.
        """

        # ---------------------------------------------------------
        # Agent 6: Verifier
        # ---------------------------------------------------------
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

            # Go backward for evidence revision.
            return RunState.DRAFTING

        # ---------------------------------------------------------
        # Agent 7: Proof-of-Ability Generator
        # ---------------------------------------------------------
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
                            "task_plan": plan,
                            "mentor_decision": mentor,
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

        # ---------------------------------------------------------
        # Agent 8: Connector
        # ---------------------------------------------------------
        recommendation = call(
            settings=ctx.settings,
            budget=ctx.budget,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Recommend a suitable next opportunity based "
                        "only on the verified Proof-of-Ability record."
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
