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
OPPORTUNITIES = Path(__file__).parent / "opportunities.json"

# This is a domain revision limit, not an API-credit limit.
MAX_REVISIONS = 1


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def load_opportunities() -> list[dict]:
    return json.loads(OPPORTUNITIES.read_text(encoding="utf-8"))


def rank_opportunities(proof: ProofOfAbility, limit: int = 3) -> list[dict]:
    """Retrieve a small candidate set from the local opportunity catalog.

    This is deliberately deterministic retrieval. The model may choose among
    retrieved records, but it is not allowed to invent an opportunity.
    """
    text = (
        f"{proof.capability} {proof.contribution} "
        + " ".join(proof.evidence)
    ).lower()

    scored = []
    for item in load_opportunities():
        score = 0
        for capability in item.get("required_capabilities", []):
            if capability.lower() in text:
                score += 2
            else:
                words = [w for w in capability.lower().split() if len(w) > 3]
                score += sum(1 for word in words if word in text)
        if item.get("type", "").lower() in text:
            score += 1
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]



def fallback_team_and_plan(project: dict, candidates: list[dict]) -> tuple[dict, dict]:
    """Deterministic profile-to-task matching used only for safe live-run recovery."""
    tasks = project.get("tasks") or []
    assignments: dict[str, dict] = {}

    def score(profile: dict, task: str) -> int:
        task_text = task.lower()
        skills = " ".join(profile.get("skills") or []).lower()
        interests = " ".join(profile.get("interests") or []).lower()
        role = (profile.get("preferred_role") or "").lower()
        bio = (profile.get("bio") or "").lower()
        text = " ".join([skills, interests, role, bio])

        score_value = 0
        frontend = ["react", "frontend", "html", "css", "javascript", "ui", "web", "layout", "interface"]
        research = ["research", "data", "python", "analysis", "source", "organize", "identify", "report"]

        if any(word in task_text for word in ["research", "identify", "sources", "data", "analysis"]):
            score_value += sum(word in text for word in research) * 4
            score_value += 4 if any(word in role for word in ["research", "data"]) else 0

        if any(word in task_text for word in ["html", "css", "react", "frontend", "layout", "responsive", "interface", "web application"]):
            score_value += sum(word in text for word in frontend) * 4
            score_value += 4 if any(word in role for word in ["frontend", "developer", "ui", "web"]) else 0

        if "mock data" in task_text or "data set" in task_text:
            score_value += sum(word in text for word in ["python", "data", "analysis", "research"]) * 4
            score_value += 4 if any(word in role for word in ["research", "data"]) else 0

        for skill in profile.get("skills") or []:
            skill_words = [token for token in skill.lower().split() if len(token) > 2]
            score_value += sum(token in task_text for token in skill_words) * 2

        return score_value

    for task in tasks:
        ranked = sorted(
            ((score(profile, task), profile) for profile in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked and ranked[0][0] > 0:
            assignments[task] = ranked[0][1]

    members: dict[str, dict] = {}
    for task, profile in assignments.items():
        name = profile.get("student_name", "Student")
        member = members.setdefault(
            name,
            {
                "student_name": name,
                "role": profile.get("preferred_role") or "Contributor",
                "reason": "Matched from the student's saved skills, interests, preferred role, and bio.",
                "matched_capabilities": list(profile.get("skills") or []),
                "assigned_tasks": [],
                "match_strength": 85,
            },
        )
        if task not in member["assigned_tasks"]:
            member["assigned_tasks"].append(task)

    team = {
        "members": list(members.values()),
        "unresolved_gaps": [task for task in tasks if task not in assignments],
    }
    plan = {
        "tasks": tasks,
        "owners": {
            task: assignments[task].get("student_name", "Unassigned")
            if task in assignments else "Unassigned"
            for task in tasks
        },
        "acceptance_conditions": {task: f"Complete: {task}" for task in tasks},
        "evidence_requirements": project.get("evidence_requirements") or [],
    }
    return team, plan


def build_flow(call=complete):

    def handle_drafting(ctx) -> RunState:
        """
        Build the student goal, project brief, team proposal,
        and task plan.

        On a revision, wait for fresh evidence instead of silently
        regenerating the student's work.
        """

        previous_verification = ctx.latest("verification")

        if ctx.latest("student_goal") is not None:
            if (
                previous_verification
                and previous_verification.get("status") == "REVISION_REQUIRED"
            ):
                if ctx.latest("evidence_submission") is None:
                    pending = callback.pending(ctx.store, ctx.run_id)
                    evidence_pending = any(
                        q.context.get("kind") == "evidence" for q in pending
                    )
                    if not evidence_pending:
                        callback.ask(
                            ctx.store,
                            ctx.run_id,
                            (
                                "Verification found missing evidence. Submit the "
                                "links, file names, screenshots, or notes that "
                                "prove what you actually completed."
                            ),
                            {
                                "kind": "evidence",
                                "resume_state": RunState.DRAFTING.value,
                                "reason": (
                                    "The verifier rejected the first evidence "
                                    "submission and needs concrete proof before "
                                    "the capability can be verified."
                                ),
                            },
                            ctx.settings,
                        )
                    return RunState.AWAITING_EXPERT

                return RunState.GATING

            return RunState.PROBING

        run_input = ctx.latest("input") or {}
        student_profile = run_input.get("student_profile", {})
        candidates = run_input.get("candidate_profiles", [])

        # ---------------------------------------------------------
        # Agent 1: Intake
        # ---------------------------------------------------------
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
                    "content": json.dumps(
                        {
                            "challenge": {
                                "title": run_input.get("title", ""),
                                "description": run_input.get("text", ""),
                            },
                            "student_profile": student_profile,
                        },
                        indent=2,
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
                    "content": json.dumps(
                        {
                            "challenge": {
                                "title": run_input.get("title", ""),
                                "description": run_input.get("text", ""),
                            },
                            "student_goal": student.model_dump(),
                        },
                        indent=2,
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
                        "Act as a semantic capability matcher. For each proposed "
                        "team member, compare the project's required capabilities "
                        "and concrete tasks against the supplied student profile: "
                        "skills, interests, experience, preferred role, availability, "
                        "and evidence links. Match by meaning and related capability, "
                        "not only exact keyword overlap. Assign only tasks that fit "
                        "the student's demonstrated profile. Never invent a student, "
                        "skill, experience, availability, or evidence. Do not match "
                        "a student merely because they are available. Explain why "
                        "the student's profile supports the role, list the exact "
                        "matched capabilities, list the tasks assigned to them, and "
                        "give a 0-100 match_strength based only on the provided data."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "project": project.model_dump(),
                            "candidate_profiles": candidates,
                        },
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
                        "Create a project plan with task owners, acceptance "
                        "conditions, and evidence requirements. Use only the "
                        "proposed team and project tasks."
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

    def handle_probing(ctx) -> RunState:
        """
        Human-in-the-loop stage.

        The first time this stage runs, it creates a real persisted
        mentor question and suspends the workflow.

        After the mentor answers, the callback system wakes the run.
        The answer is then converted into a structured mentor_decision
        record that later agents can read.
        """

        if ctx.latest("mentor_decision") is not None:
            return RunState.GATING

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
                    "kind": "mentor",
                    "resume_state": RunState.PROBING.value,
                    "options": [
                        "Prioritise one unified student experience",
                        "Keep every existing channel unchanged",
                    ],
                    "reason": (
                        "The system detected a possible conflict between a "
                        "unified discovery experience and existing club channels."
                    ),
                },
                ctx.settings,
            )
            return RunState.AWAITING_EXPERT

        answer = answers[-1].payload

        # A project created before profile-refresh support may have been matched
        # without the creator. When the mentor resumes that run, refresh the
        # team and task plan using the latest candidate profiles.
        run_input = ctx.latest("input") or {}
        candidates = run_input.get("candidate_profiles") or []
        current_creator = (
            run_input.get("created_by")
            or ctx.store.meta(ctx.run_id).get("created_by")
        )
        team = ctx.latest("team_proposal") or {}
        member_names = {
            member.get("student_name")
            for member in (team.get("members") or [])
            if isinstance(member, dict)
        }
        creator_name = next(
            (
                profile.get("student_name")
                for profile in candidates
                if profile.get("email") == current_creator
            ),
            None,
        )

        if creator_name and creator_name not in member_names:
            project = ctx.latest("project_brief")
            try:
                refreshed_team = call(
                    settings=ctx.settings,
                    budget=ctx.budget,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Act as a semantic capability matcher. Compare the "
                                "project requirements and concrete tasks against every "
                                "supplied student profile. Include suitable students "
                                "who can genuinely contribute, including the project "
                                "creator when their profile fits. Assign concrete tasks "
                                "only when supported by the profile. Never invent skills "
                                "or experience."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"project": project, "candidate_profiles": candidates},
                                indent=2,
                            ),
                        },
                    ],
                    schema=TeamProposal,
                    step="team_refresh",
                )
                refreshed_plan = call(
                    settings=ctx.settings,
                    budget=ctx.budget,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Create a project plan with task owners, acceptance "
                                "conditions, and evidence requirements using only "
                                "the project tasks and proposed team."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "project": project,
                                    "team": refreshed_team.model_dump(),
                                },
                                indent=2,
                            ),
                        },
                    ],
                    schema=TaskPlan,
                    step="plan_refresh",
                )
                ctx.append(
                    "team_proposal",
                    refreshed_team.model_dump(),
                    produced_by="agent:team_matcher_refresh",
                )
                ctx.append(
                    "task_plan",
                    refreshed_plan.model_dump(),
                    produced_by="agent:orchestrator_refresh",
                )
            except Exception:
                fallback_team, fallback_plan = fallback_team_and_plan(project, candidates)
                ctx.append(
                    "team_proposal",
                    fallback_team,
                    produced_by="system:team_refresh_fallback",
                )
                ctx.append(
                    "task_plan",
                    fallback_plan,
                    produced_by="system:plan_refresh_fallback",
                )

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
        Verify real submitted evidence, then create Proof-of-Ability and
        recommend an opportunity from the local opportunity catalog.
        """

        project = ctx.latest("project_brief")
        plan = ctx.latest("task_plan")
        mentor = ctx.latest("mentor_decision")
        evidence = ctx.latest("evidence_submission")

        # Do not spend a verifier model call when no real evidence exists yet.
        # The human must submit concrete proof first.
        if evidence is None:
            pending = callback.pending(ctx.store, ctx.run_id)
            evidence_pending = any(
                q.context.get("kind") == "evidence" for q in pending
            )
            if not evidence_pending:
                callback.ask(
                    ctx.store,
                    ctx.run_id,
                    (
                        "Submit concrete evidence for the completed project work: "
                        "links, screenshots, file names, prototype URLs, notes, "
                        "or other artifacts showing what was actually completed."
                    ),
                    {
                        "kind": "evidence",
                        "resume_state": RunState.GATING.value,
                        "reason": (
                            "Verification cannot responsibly pass without "
                            "evidence of the work."
                        ),
                    },
                    ctx.settings,
                )
            return RunState.AWAITING_EXPERT

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
                            "evidence_submission": evidence,
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
                            "The project did not pass verification within "
                            "the revision limit."
                        ),
                    },
                    produced_by="system",
                )
                return RunState.FAILED

            return RunState.DRAFTING

        # ---------------------------------------------------------
        # Agent 7: Proof-of-Ability Generator
        # ---------------------------------------------------------
        student_profile = (ctx.latest("input") or {}).get("student_profile") or {}
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
                            "student_profile": student_profile,
                            "project": project,
                            "task_plan": plan,
                            "mentor_decision": mentor,
                            "verification": result.model_dump(),
                            "evidence_submission": evidence,
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
        candidates = rank_opportunities(proof)

        recommendation = call(
            settings=ctx.settings,
            budget=ctx.budget,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the ImpactLoop Connector. Recommend exactly "
                        "one next opportunity from the candidate catalog below. "
                        "Do not invent an opportunity, organisation, or "
                        "requirement. Explain the match using only the verified "
                        "Proof-of-Ability record."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "proof_of_ability": proof.model_dump(),
                            "candidate_opportunities": candidates,
                        },
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
