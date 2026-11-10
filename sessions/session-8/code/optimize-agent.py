"""Optimizing an agent (§7): BootstrapFewShot on the agentic inbox.

Session 5 said optimizers can tune an agent like any other program because
ReAct is modules all the way down. This is the first time the series runs
one. The optimizer runs the agent over the training emails, calls the inbox
metric with the trace, and keeps every run all four checks accept. For the
`draft.react` predictor, an accepted run is a whole trajectory: the demo
shows the model a worked search.

max_labeled_demos is 0: the training rows carry labels for triage but no
reply and no trajectory, so there is nothing to copy into the draft
predictors by hand.
"""

import random

import dspy

from common import (JUDGE_LM, STATE, TASK_LM, AgenticInbox, cost, evaluate_inbox,
                    inbox_metric, load_emails, print_breakdown)

dspy.configure(lm=TASK_LM)

trainset, testset = load_emails("train"), load_emails("test")
random.Random(3).shuffle(trainset)

verdicts = []


def watched_metric(example, pred, trace=None):
    verdict = inbox_metric(example, pred, trace)
    verdicts.append((example.email, trace is not None, verdict))
    return verdict


optimizer = dspy.BootstrapFewShot(
    metric=watched_metric,
    max_bootstrapped_demos=2,
    max_labeled_demos=0,
    max_rounds=1,
)
optimized = optimizer.compile(AgenticInbox(max_iters=4), trainset=trainset)

print("--- metric calls during compile ---")
for email, traced, verdict in verdicts:
    print(f"trace={'yes' if traced else 'no ':3s} {str(verdict):5s} {email[:60]}")
print(f"{len(verdicts)} calls, {sum(1 for *_, v in verdicts if v)} accepted")

print("\n--- what the optimizer wrote, per predictor ---")
for name, predictor in optimized.named_predictors():
    sizes = [len(str(dict(d))) for d in predictor.demos]
    print(f"{name:22s} demos={len(predictor.demos)}  chars={sizes}")

print("\n--- one demo for draft.react: a worked trajectory ---")
react_demos = optimized.draft.react.demos
if react_demos:
    demo = dict(react_demos[0])
    for key in ("email", "category", "trajectory", "next_thought", "next_tool_name", "next_tool_args"):
        if key in demo:
            text = str(demo[key]).replace("\n", " ")
            print(f"{key:15s}: {text[:200]}{'...' if len(text) > 200 else ''}")
else:
    print("(no trajectory demo was accepted; see the metric calls above)")

print("\n--- test split, before and after ---")
n = len(testset)
for label, program in [("before", AgenticInbox(max_iters=4)), ("after", optimized)]:
    score, totals, _ = evaluate_inbox(program, testset)
    print_breakdown(label, score, totals, n)

STATE.mkdir(exist_ok=True)
optimized.save(STATE / "agentic-inbox-bootstrap.json")
print(f"\nsaved to {STATE.name}/agentic-inbox-bootstrap.json")
print(f"cost : task ${cost(TASK_LM):.4f}  judge ${cost(JUDGE_LM):.4f}")
