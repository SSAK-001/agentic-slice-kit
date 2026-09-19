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

        if ctx.latest("student_goal") is not None:
            # Once the mentor has made the decision, later passes of DRAFTING
            # are only routing back into verification. Never reopen the mentor
            # checkpoint just because another task needs evidence.
            if ctx.latest("mentor_decision") is not None:
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
        Execute student work one assigned task at a time.

        Each task has its own evidence submission and verifier decision.
        The workflow never verifies the same submission twice. Once every
        assigned task is verified, each participating student gets their own
        Proof-of-Ability and next-opportunity recommendation.
        """
        project = ctx.latest("project_brief") or {}
        plan = ctx.latest("task_plan") or {}
        mentor = ctx.latest("mentor_decision") or {}
        run_input = ctx.latest("input") or {}
        candidates = run_input.get("candidate_profiles") or []

        tasks = plan.get("tasks") or []
        owners = plan.get("owners") or {}
        conditions = plan.get("acceptance_conditions") or {}

        # Support older runs that predate task-level evidence.
        if not tasks:
            evidence = ctx.latest("evidence_submission")
            if evidence is None:
                pending = callback.pending(ctx.store, ctx.run_id)
                if not any(q.context.get("kind") == "evidence" for q in pending):
                    callback.ask(
                        ctx.store,
                        ctx.run_id,
                        (
                            "Submit concrete evidence for the completed project work: "
                            "links, screenshots, file names, prototype URLs, notes, "
                            "or other artifacts showing what was actually completed."
                        ),
                        {"kind": "evidence", "resume_state": RunState.GATING.value},
                        ctx.settings,
                    )
                return RunState.AWAITING_EXPERT

            result = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {"role": "system", "content": load_prompt("verify")},
                    {"role": "user", "content": json.dumps(
                        {
                            "project": project,
                            "task_plan": plan,
                            "mentor_decision": mentor,
                            "evidence_submission": evidence,
                        },
                        indent=2,
                    )},
                ],
                schema=VerificationResult,
                step="verify",
            )
            ctx.append("verification", result.model_dump(), produced_by="agent:verifier")
            if result.status == "REVISION_REQUIRED":
                return RunState.DRAFTING

            student_profile = run_input.get("student_profile") or {}
            proof = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {"role": "system", "content": load_prompt("proof")},
                    {"role": "user", "content": json.dumps(
                        {
                            "student_profile": student_profile,
                            "project": project,
                            "task_plan": plan,
                            "mentor_decision": mentor,
                            "verification": result.model_dump(),
                            "evidence_submission": evidence,
                        },
                        indent=2,
                    )},
                ],
                schema=ProofOfAbility,
                step="proof",
            )
            ctx.append("proof_of_ability", proof.model_dump(), produced_by="agent:proof")
            candidates_for_opportunity = rank_opportunities(proof)
            recommendation = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the ImpactLoop Connector. Recommend exactly "
                            "one next opportunity from the candidate catalog below. "
                            "Do not invent an opportunity, organisation, or requirement."
                        ),
                    },
                    {"role": "user", "content": json.dumps(
                        {
                            "proof_of_ability": proof.model_dump(),
                            "candidate_opportunities": candidates_for_opportunity,
                        },
                        indent=2,
                    )},
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

        def candidates_for_owner(owner: str) -> list[dict]:
            return [
                profile for profile in candidates
                if profile.get("student_name") == owner
            ]

        def owner_email(owner: str) -> str:
            matches = candidates_for_owner(owner)
            return matches[0].get("email", "") if matches else ""

        task_submissions = ctx.history("task_evidence_submission")
        task_verifications = ctx.history("task_verification")

        def records_for(kind_records, task):
            return [
                record.payload
                for record in kind_records
                if record.payload.get("task") == task
            ]

        verified_tasks = {
            payload.get("task")
            for record in task_verifications
            for payload in [record.payload]
            if payload.get("status") == "PASS"
        }

        assigned = [
            task for task in tasks
            if isinstance(owners, dict) and owners.get(task)
            and owners.get(task) not in {"Unassigned", "None"}
        ]

        if not assigned:
            ctx.append(
                "failure",
                {
                    "kind": "no_assigned_tasks",
                    "detail": "The project plan contains no tasks with valid student owners.",
                },
                produced_by="system",
            )
            return RunState.FAILED

        # Create a separate evidence request for every assigned task that
        # has not yet been submitted or verified. This lets different students
        # work in parallel instead of blocking the whole team on the first task.
        pending = callback.pending(ctx.store, ctx.run_id)

        for task in assigned:
            owner = owners[task]
            submissions = records_for(task_submissions, task)
            verifications = records_for(task_verifications, task)
            latest_verification = verifications[-1] if verifications else None

            if latest_verification and latest_verification.get("status") == "PASS":
                continue

            needs_submission = (
                not submissions
                or len(submissions) <= len(verifications)
            )
            if not needs_submission:
                continue

            task_pending = next(
                (
                    q for q in pending
                    if q.context.get("kind") == "task_evidence"
                    and q.context.get("task") == task
                ),
                None,
            )
            if task_pending is not None:
                continue

            missing = (
                (latest_verification or {}).get("missing_evidence") or []
            )
            missing_text = ", ".join(missing)
            revision = bool(latest_verification)

            question = (
                f"Submit evidence for your task: {task}. "
                f"Acceptance condition: {conditions.get(task, 'Show concrete work completed.')}"
            )
            if revision and missing_text:
                question += f" The verifier still needs: {missing_text}."

            callback.ask(
                ctx.store,
                ctx.run_id,
                question,
                {
                    "kind": "task_evidence",
                    "task": task,
                    "owner": owner,
                    "owner_email": owner_email(owner),
                    "acceptance_condition": conditions.get(
                        task,
                        "Show concrete work completed.",
                    ),
                    "missing_evidence": missing,
                    "resume_state": RunState.GATING.value,
                    "reason": (
                        "Student evidence is required for this assigned task."
                        if not revision
                        else "The previous evidence did not satisfy verification; submit a stronger revision."
                    ),
                },
                ctx.settings,
            )

        # If any assigned task is still waiting for evidence, keep the run
        # suspended. The questions are independently answerable by their owners.
        remaining_work = []
        for task in assigned:
            submissions = records_for(ctx.history("task_evidence_submission"), task)
            verifications = records_for(ctx.history("task_verification"), task)
            latest_verification = verifications[-1] if verifications else None
            if not latest_verification or latest_verification.get("status") != "PASS":
                remaining_work.append(task)

        if remaining_work:
            return RunState.AWAITING_EXPERT

        # Verify each newly submitted task. If a task fails, its owner
        # will receive a revision request on the next pass.
        submitted_unverified = []
        for task in assigned:
            owner = owners[task]
            submissions = records_for(ctx.history("task_evidence_submission"), task)
            verifications = records_for(ctx.history("task_verification"), task)
            latest_submission = submissions[-1] if submissions else None
            latest_verification = verifications[-1] if verifications else None

            if not latest_submission:
                continue
            if latest_verification is not None and len(verifications) >= len(submissions):
                continue

            submitted_unverified.append((task, owner, latest_submission))

        if submitted_unverified:
            for task, owner, evidence in submitted_unverified:
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
                                    "student_profile": next(
                                        (
                                            profile for profile in candidates
                                            if profile.get("student_name") == owner
                                        ),
                                        {},
                                    ),
                                    "project": project,
                                    "task": task,
                                    "task_owner": owner,
                                    "acceptance_condition": conditions.get(task, ""),
                                    "project_evidence_requirements": (
                                        project.get("evidence_requirements") or []
                                    ),
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
                payload = result.model_dump()
                payload.update({"task": task, "task_owner": owner})
                ctx.append("task_verification", payload, produced_by="agent:verifier")

                if result.status == "REVISION_REQUIRED":
                    return RunState.DRAFTING

            # Re-evaluate the task set. Any remaining tasks will either have
            # a pending evidence request or will be verified on a future pass.
            return RunState.DRAFTING

        # All task verifications have passed. Continue to Proof-of-Ability.
        # Every assigned task has passed. Produce one proof and opportunity for
        # each participating student, using only that student's verified work.
        proof_history = ctx.history("proof_of_ability")
        recommendation_history = ctx.history("opportunity_recommendation")
        existing_proof_owners = {
            record.payload.get("student_name")
            for record in proof_history
        }
        existing_recommendation_owners = {
            record.payload.get("student_name")
            for record in recommendation_history
        }

        owners_in_order = []
        for task in assigned:
            owner = owners[task]
            if owner not in owners_in_order:
                owners_in_order.append(owner)

        for owner in owners_in_order:
            if owner in existing_proof_owners and owner in existing_recommendation_owners:
                continue

            student_tasks = [task for task in assigned if owners[task] == owner]
            student_submissions = [
                record.payload
                for record in task_submissions
                if record.payload.get("task") in student_tasks
            ]
            student_verifications = [
                record.payload
                for record in task_verifications
                if record.payload.get("task") in student_tasks
                and record.payload.get("status") == "PASS"
            ]
            student_profile = next(
                (
                    profile for profile in candidates
                    if profile.get("student_name") == owner
                ),
                {},
            )

            proof = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {"role": "system", "content": load_prompt("proof")},
                    {"role": "user", "content": json.dumps(
                        {
                            "student_profile": student_profile,
                            "student_name": owner,
                            "project": project,
                            "task_plan": plan,
                            "mentor_decision": mentor,
                            "verified_tasks": student_verifications,
                            "evidence_submissions": student_submissions,
                        },
                        indent=2,
                    )},
                ],
                schema=ProofOfAbility,
                step="proof",
            )
            ctx.append("proof_of_ability", proof.model_dump(), produced_by="agent:proof")

            opportunities = rank_opportunities(proof)
            recommendation = call(
                settings=ctx.settings,
                budget=ctx.budget,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the ImpactLoop Connector. Recommend exactly "
                            "one next opportunity from the candidate catalog below. "
                            "Do not invent an opportunity, organisation, or requirement. "
                            "Choose only an opportunity supported by the verified proof."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "student_name": owner,
                                "proof_of_ability": proof.model_dump(),
                                "candidate_opportunities": opportunities,
                            },
                            indent=2,
                        ),
                    },
                ],
                schema=OpportunityRecommendation,
                step="connector",
            )
            recommendation_payload = recommendation.model_dump()
            recommendation_payload["student_name"] = owner
            ctx.append(
                "opportunity_recommendation",
                recommendation_payload,
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
