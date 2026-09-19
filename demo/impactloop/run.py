from demo.impactloop.flow import build_flow
from demo.impactloop.stub import Stub

from slice import runner
from slice.config import settings
from slice.store import Store


store = Store(":memory:")

run_id = store.create_run(
    "impactloop",
    {
        "description": "ImpactLoop MVP fake workflow",
    },
)

store.append(
    run_id,
    "input",
    {
        "text": (
            "I know Python and basic data analysis, "
            "but I do not have a real project that proves my ability."
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
print("Run ID:", run_id)
print("Final state:", final_state)
print()

for kind in [
    "input",
    "student_goal",
    "project_brief",
    "verification",
    "proof_of_ability",
    "failure",
]:
    records = store.history(run_id, kind)
    for record in records:
        print(f"[{kind}] {record.produced_by}")
        print(record.payload)
        print("-" * 50)
