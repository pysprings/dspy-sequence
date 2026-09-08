"""Reading and shipping an optimized program (§7).

Three things a saved state lets you do: read what the optimizer wrote, load
it into a fresh instance of the same architecture (Session 5's roundtrip),
and run that instance under a different task model. The last one is the
transfer test: an instruction written while gpt-4.1-nano was the task model,
evaluated with gpt-5.6-luna instead.
"""

import json

import dspy

from common import (DEPLOY_LM, STATE, TASK_LM, SupportInbox, UnixAssistant,
                    command_match, cost, inbox_metric, load_emails, load_unix)

dspy.configure(lm=TASK_LM)


def describe(program):
    for name, pred in program.named_predictors():
        print(f"  {name:14s} demos={len(pred.demos):<2d} "
              f"instructions={pred.signature.instructions[:70]!r}...")


print("--- what each saved state holds ---")
for label, cls, path in [("bootstrap", UnixAssistant, "unix-bootstrap.json"),
                         ("gepa unix", UnixAssistant, "unix-gepa.json"),
                         ("gepa inbox", SupportInbox, "inbox-gepa.json")]:
    program = cls()
    program.load(STATE / path)
    print(f"{label}:")
    describe(program)

print("\n--- where it lives in the file ---")
state = json.loads((STATE / "unix-gepa.json").read_text())
print(f"top-level keys        : {list(state)}")
print(f"predict keys          : {list(state['predict'])}")
print(f"instructions at       : state['predict']['signature']['instructions']")
print(f"demos at              : state['predict']['demos']  ({len(state['predict']['demos'])} entries)")
print(f"dspy version recorded : {state['metadata']['dependency_versions']['dspy']}")

print("\n--- transfer test: same state, different task model ---")
unix_test, inbox_test = load_unix("test"), load_emails("test")
unix_eval = dspy.Evaluate(devset=unix_test, metric=command_match, num_threads=4)
inbox_eval = dspy.Evaluate(devset=inbox_test, metric=inbox_metric, num_threads=3)

unix_gepa, inbox_gepa = UnixAssistant(), SupportInbox()
unix_gepa.load(STATE / "unix-gepa.json")
inbox_gepa.load(STATE / "inbox-gepa.json")

rows = []
for lm_name, lm in [("gpt-4.1-nano (optimized on)", TASK_LM), ("gpt-5.6-luna (never seen)", DEPLOY_LM)]:
    with dspy.context(lm=lm):
        rows.append((lm_name,
                     unix_eval(UnixAssistant()).score, unix_eval(unix_gepa).score,
                     inbox_eval(SupportInbox()).score, inbox_eval(inbox_gepa).score))
print(f"{'task model':28s} {'unix base':>10s} {'unix gepa':>10s} {'inbox base':>11s} {'inbox gepa':>11s}")
for name, ub, ug, ib, ig in rows:
    print(f"{name:28s} {ub:>10.2f} {ug:>10.2f} {ib:>11.2f} {ig:>11.2f}")

print(f"\ncost : nano ${cost(TASK_LM):.3f}  luna ${cost(DEPLOY_LM):.3f}")
