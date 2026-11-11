"""BestOfN (§4): draft up to N times, keep the best reply.

The draft step of the grounded inbox is wrapped in `dspy.BestOfN`. Each
attempt is a fresh sample from the same prompt. The reward function scores
every attempt. The loop stops at the first attempt that reaches the
threshold, or returns the highest-scoring attempt after N tries.

Nothing in the prompt changes between attempts. That is the whole idea, and
the output shows its limit.
"""

import json

import dspy

from common import (STATE, TASK_LM, GroundedInbox, load_emails, print_attempt_grid, print_result,
                    reply_reward, run_test, spent, with_retry, wrapped)

dspy.configure(lm=TASK_LM)

# The two lines that matter. `with_retry` in common.py does the same thing
# and also records every attempt, so the script can print them.
inbox = GroundedInbox()
inbox.draft = dspy.BestOfN(module=inbox.draft, N=3, reward_fn=reply_reward, threshold=1.0)

print("--- the predictor tree, with and without the loop ---")
print(f"GroundedInbox : {[name for name, _ in GroundedInbox().named_predictors()]}")
print(f"with BestOfN  : {[name for name, _ in inbox.named_predictors()]}")

best, recorder = with_retry(dspy.BestOfN, N=3, threshold=1.0)


def show_attempts(email: str):
    pred = best(email=email)
    print(f"email : {email}")
    for i, (reward, reply) in enumerate(recorder.attempts[email], 1):
        print(wrapped(f"  attempt {i}  reward {reward:.2f}  ", reply))
    print(wrapped("  sent  : ", pred.reply))


print("\n--- an email where a second sample is enough ---")
show_attempts("Minor thing: the settings page misspells 'preferences' in the sidebar.")

print("\n--- an email where three samples are not enough ---")
show_attempts("I was charged twice for my March invoice. Please refund the duplicate.")

print("\n--- test split: one draft against BestOfN ---")
testset = load_emails("test")
recorder.attempts.clear()
base = run_test(GroundedInbox(), testset)
loop = run_test(best, testset, recorder)
print_result("one draft", base, header=True)
print_result("BestOfN, N=3", loop)
print(f"attempts per email : one draft {base['attempts']:.1f}   BestOfN {loop['attempts']:.1f}")

print("\n--- the attempts, per email ---")
print_attempt_grid(loop["rows"], recorder)
saved = []
for example, pred, _, checks in loop["rows"]:
    attempts = recorder.attempts[example.email]
    saved.append({"email": example.email,
                  "attempts": [{"reward": round(r, 2), "reply": reply} for r, reply in attempts],
                  "sent": pred.reply, "checks": checks})

print("\n--- the reply it sent for the praise email ---")
praise = next(row for row in saved if row["email"].startswith("Just wanted to say"))
print(wrapped("", praise["sent"]))

STATE.mkdir(exist_ok=True)
(STATE / "best-of-n-attempts.json").write_text(json.dumps(saved, indent=2))
print(f"\nsaved every attempt to {STATE.name}/best-of-n-attempts.json")
print(f"cost : ${spent():.4f}")

assert [name for name, _ in inbox.named_predictors()] == ["triage", "extract", "draft.module.predict"]
assert all(len(row["attempts"]) <= 3 for row in saved)
assert loop["attempts"] > base["attempts"]
