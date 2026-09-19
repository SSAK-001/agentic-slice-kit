from demo.impactloop.flow import build_flow
from demo.impactloop.stub import Stub

from slice import runner
from slice.config import settings
from slice.store import Store


store = Store(":memory:")

run_id = store.create_run(
    "impactloop",
    {
        "description": "ImpactLoop MVP - Student Event Discovery",
    },
)

store.append(
    run_id,
    "input",
    {
        "text": (
            "Students miss useful campus events and opportunities "
            "because information is scattered across WhatsApp groups, "
            "posters, club pages, and separate channels."
        )
    },
    "user",
)

final_state = runner.advance(
    store,
    run_id,
    build_flow(call=Stub()),
    settings,
)

print("\n==============================")
print("IMPACTLOOP DEMO")
print("==============================")
print("Challenge: Student Event Discovery")
print("Run ID:", run_id)
print("Final state:", final_state)
print()

for kind in [
    "input",
    "student_goal",
    "team_proposal",
    "task_plan",
    "mentor_decision",
    "project_brief",
    "verification",
    "proof_of_ability",
    "opportunity_recommendation",
    "failure",
]:
    records = store.history(run_id, kind)

    for record in records:
        print(f"[{kind}] {record.produced_by}")
        print(record.payload)
        print("-" * 50)
