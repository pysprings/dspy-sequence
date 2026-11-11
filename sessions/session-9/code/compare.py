"""The measurement (§6): four programs, three runs each, and a second judge.

Six test emails make a noisy score. One changed check moves it 4.2 points,
and every draft is a random sample. So this script runs each program three
times. A run is a fresh set of samples: `rollout_id` changes the cache key,
so the same prompt is sampled again.

The four programs:
  one draft            the grounded inbox as Session 8 left it
  BestOfN              up to three attempts, keep the best
  Refine, task-model hints     up to three attempts, the task model writes the hints
  Refine, larger-model hints   up to three attempts, the larger model writes the hints

The reward and the metric use the same judge model. A loop that learns to
please that judge would raise the score without better replies. So the last
step gives run 1's final replies to a second judge from another vendor.
"""

import json

import dspy

from common import (ACTIONABLE_Q, GROUNDED_Q, HINT_LM, SECOND_JUDGE, STATE, TASK_LM, GroundedInbox,
                    assess, load_emails, run_test, spent, with_retry)

dspy.configure(lm=TASK_LM)
testset = load_emails("test")
n = len(testset)

PROGRAMS = [   # label, wrapper, N, model that writes hints
    ("one draft", None, 1, None),
    ("BestOfN", dspy.BestOfN, 3, None),
    ("Refine, task-model hints", dspy.Refine, 3, TASK_LM),
    ("Refine, larger-model hints", dspy.Refine, 3, HINT_LM),
]
RUNS = [0, 100, 200]   # the rollout_id each run starts from

results = {label: [] for label, *_ in PROGRAMS}
for start in RUNS:
    task_lm = TASK_LM.copy(rollout_id=start) if start else TASK_LM
    for label, wrapper, tries, hint_lm in PROGRAMS:
        if wrapper is None:                      # the plain program, with no loop and no reward
            program, recorder = GroundedInbox(), None
            program.set_lm(task_lm)
        else:
            program, recorder = with_retry(wrapper, N=tries, threshold=1.0, task_lm=task_lm)
        with dspy.context(lm=hint_lm or TASK_LM):
            results[label].append(run_test(program, testset, recorder))
    print(f"run starting at rollout_id {start} done", flush=True)

print("\n--- score on the test split, three runs ---")
print(f"{'program':27s} {'run 1':>6s} {'run 2':>6s} {'run 3':>6s} {'mean':>6s}")
for label, runs in results.items():
    scores = [r["score"] for r in runs]
    print(f"{label:27s} {scores[0]:>6.1f} {scores[1]:>6.1f} {scores[2]:>6.1f} {sum(scores) / 3:>6.1f}")

print("\n--- the checks, summed over three runs (18 replies) ---")
print(f"{'program':27s} {'category':>9s} {'urgency':>8s} {'grounded':>9s} {'actionable':>11s} {'redirects':>10s}")
for label, runs in results.items():
    total = {name: sum(r["totals"][name] for r in runs) for name in runs[0]["totals"]}
    redirects = sum(r["redirects"] for r in runs)
    print(f"{label:27s} {total['category']:>6d}/18 {total['urgency']:>5d}/18 {total['grounded']:>6d}/18 "
          f"{total['actionable']:>8d}/18 {redirects:>7d}/18")

print("\n--- what one email costs, mean of three runs ---")
print(f"{'program':27s} {'attempts':>9s} {'hints':>6s} {'judge calls':>12s}")
for label, runs in results.items():
    drafts = sum(r["attempts"] for r in runs) / 3
    hints = sum(r["hints"] for r in runs) / 3
    judged = sum(r["judged"] for r in runs) / 3
    print(f"{label:27s} {drafts:>9.1f} {hints:>6.1f} {judged:>12.1f}")
print("(one judge call is one question; the reward asks two for each attempt that is")
print(" not a redirect; the calls of triage, extract, and the metric are the same for")
print(" every program and are left out)")

print("\n--- run 1's final replies, scored by a second judge ---")


def second_opinion(rows) -> tuple[int, int]:
    grounded = actionable = 0
    with dspy.context(lm=SECOND_JUDGE):
        for example, pred, _, _ in rows:
            grounded += assess(email=example.email, reply=pred.reply, question=GROUNDED_Q).verdict
            actionable += assess(email=example.email, reply=pred.reply, question=ACTIONABLE_Q).verdict
    return grounded, actionable


print(f"{'program':27s} {'first judge':>14s} {'second judge':>14s}   (of 12 judged checks)")
for label, runs in results.items():
    first = runs[0]["totals"]["grounded"] + runs[0]["totals"]["actionable"]
    second = sum(second_opinion(runs[0]["rows"]))
    print(f"{label:27s} {first:>11d}/12 {second:>11d}/12")

saved = {
    label: [{"score": r["score"], "totals": r["totals"], "redirects": r["redirects"],
             "attempts_per_email": r["attempts"], "hints_per_email": r["hints"],
             "replies": [{"email": ex.email, "reply": pred.reply} for ex, pred, _, _ in r["rows"]]}
            for r in runs]
    for label, runs in results.items()
}
STATE.mkdir(exist_ok=True)
(STATE / "compare.json").write_text(json.dumps(saved, indent=2))
print(f"\nsaved all 72 final replies to {STATE.name}/compare.json")
print(f"cost : ${spent():.4f}")

mean = {label: sum(r["score"] for r in runs) / 3 for label, runs in results.items()}
assert mean["Refine, larger-model hints"] > mean["one draft"]
assert all(r["totals"]["category"] == n for runs in results.values() for r in runs), "a label moved"
