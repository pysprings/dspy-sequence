"""Baseline first (§3): the numbers every later section is compared against.

Both programs, unoptimized, scored on their held-out test splits with the
Session 6 metrics. The optimizer never sees the test split. The cost lines
use each LM's call history; cached calls cost nothing, so a rerun reports
less than the first run.
"""

import dspy

from common import (JUDGE_LM, TASK_LM, SupportInbox, UnixAssistant, command_match,
                    cost, exact_match, inbox_metric, load_emails, load_unix)

dspy.configure(lm=TASK_LM)

print("--- data ---")
for name, loader in [("unix", load_unix), ("emails", load_emails)]:
    counts = {split: len(loader(split)) for split in ("train", "val", "test")}
    print(f"{name:7s}: {counts}")

print("\n--- unix baseline (test split, three-tier metric) ---")
unix_test = load_unix("test")
evaluator = dspy.Evaluate(devset=unix_test, metric=command_match, num_threads=4)
result = evaluator(UnixAssistant())
print(f"score : {result.score}")
for example, prediction, score in result.results:
    if not exact_match(example.command, prediction.command):
        print(f"  {score:>4}  gold: {example.command!r:24s} predicted: {prediction.command!r}")

print("\n--- inbox baseline (test split, composite judge metric) ---")
inbox_test = load_emails("test")
evaluator = dspy.Evaluate(devset=inbox_test, metric=inbox_metric, num_threads=3)
result = evaluator(SupportInbox())
print(f"score : {result.score}")
for example, prediction, score in result.results:
    print(f"  {score:>5}  {example.category:>10}/{example.urgency:<4}  "
          f"got {prediction.category}/{prediction.urgency}  | {example.email[:44]}...")

print("\n--- cost of this script ---")
print(f"task model  : ${cost(TASK_LM):.4f}")
print(f"judge model : ${cost(JUDGE_LM):.4f}")
