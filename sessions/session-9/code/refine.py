"""Refine (§5): draft, score, write a hint, draft again.

`dspy.Refine` runs the same loop as BestOfN with one addition. After an
attempt that scores below the threshold, a hidden predictor reads the
attempt, the wrapped module's source code, and the reward function's source
code. It writes a hint. The next attempt receives the hint as an extra
input field named `hint_`.

The hint writer uses whatever model `dspy.configure` names. This script
runs Refine twice: once with the task model writing its own hints, and once
with the larger model writing them.
"""

import json

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY
from dspy.predict.refine import OfferFeedback

from common import (HINT_LM, HINT_WRITER_MARK, STATE, TASK_LM, GroundedInbox, hints_written,
                    load_emails, print_attempt_grid, print_result, reply_reward, run_test, spent,
                    with_retry, wrapped)

dspy.configure(lm=TASK_LM)

# The two lines that matter. Only the class name differs from §4.
inbox = GroundedInbox()
inbox.draft = dspy.Refine(module=inbox.draft, N=3, reward_fn=reply_reward, threshold=1.0)

email = "I was charged twice for my March invoice. Please refund the duplicate."


def show_attempts(program, recorder, label: str):
    before = len(hints_written())
    pred = program(email=email)
    hints = hints_written(since=before)
    print(f"email : {email}")
    for i, (reward, reply) in enumerate(recorder.attempts[email], 1):
        print(wrapped(f"  attempt {i}  reward {reward:.2f}  ", reply))
        if reward < 1.0 and i <= len(hints):
            print(wrapped(f"  hint from the {label}: ", hints[i - 1]))
    print(wrapped("  sent  : ", pred.reply))
    return pred


print("--- what the hint writer reads ---")
for name, field in OfferFeedback.input_fields.items():
    print(f"  {name:19s} {field.json_schema_extra['desc']}")

print("\n--- Refine, with the task model writing the hints ---")
weak, weak_recorder = with_retry(dspy.Refine, N=3, threshold=1.0)
show_attempts(weak, weak_recorder, "task model")

print("\n--- Refine, with the larger model writing the hints ---")
strong, strong_recorder = with_retry(dspy.Refine, N=3, threshold=1.0)
with dspy.context(lm=HINT_LM):
    sent = show_attempts(strong, strong_recorder, "larger model")

print("\n--- the prompt of a hinted attempt gains one input field ---")
hinted = [entry for entry in GLOBAL_HISTORY
          if "[[ ## hint_ ## ]]" in entry["messages"][-1]["content"]
          and HINT_WRITER_MARK not in entry["messages"][0]["content"]]
system_prompt = hinted[-1]["messages"][0]["content"]
print("\n".join(line for line in system_prompt.splitlines() if line[:2] in ("1.", "2.", "3.", "4.")))

print("\n--- test split ---")
testset = load_emails("test")
weak_recorder.attempts.clear()
strong_recorder.attempts.clear()
base = run_test(GroundedInbox(), testset)
hints_before = len(hints_written())
weak_run = run_test(weak, testset, weak_recorder)
nano_hints = hints_written(since=hints_before)
with dspy.context(lm=HINT_LM):
    strong_run = run_test(strong, testset, strong_recorder)
print_result("one draft", base, header=True)
print_result("Refine, task-model hints", weak_run)
print_result("Refine, larger-model hints", strong_run)
print(f"attempts per email : task-model hints {weak_run['attempts']:.1f}   larger-model hints {strong_run['attempts']:.1f}")
print(f"hints per email    : task-model hints {weak_run['hints']:.1f}   larger-model hints {strong_run['hints']:.1f}")
print(f"task-model hints that say N/A : {sum(h.strip() == 'N/A' for h in nano_hints)} of {len(nano_hints)}")

print("\n--- the attempts with larger-model hints, per email ---")
print_attempt_grid(strong_run["rows"], strong_recorder)
saved = []
for example, pred, _, checks in strong_run["rows"]:
    attempts = strong_recorder.attempts[example.email]
    saved.append({"email": example.email,
                  "attempts": [{"reward": round(r, 2), "reply": reply} for r, reply in attempts],
                  "sent": pred.reply, "checks": checks})

STATE.mkdir(exist_ok=True)
(STATE / "refine-attempts.json").write_text(json.dumps(saved, indent=2))
print(f"\nsaved every attempt to {STATE.name}/refine-attempts.json")
print(f"cost : ${spent():.4f}")

assert "support@" not in sent.reply.lower(), "the stronger hint should stop the redirect"
assert strong_run["redirects"] < base["redirects"]
