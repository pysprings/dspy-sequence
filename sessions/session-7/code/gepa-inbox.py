"""GEPA on SupportInbox (§6): feedback routed to each predictor.

SupportInbox has three predictors. The metric scores the whole program, but
GEPA asks it once per predictor, passing `pred_name`, so the feedback text
can say what that predictor got wrong: label mismatches to `triage`, the
judge's failed dimensions to `draft.predict`. See common.inbox_feedback.

Session 6 found the replies were polite and promised nothing. This is the
run that targets that finding.
"""

import json

import dspy

from common import (JUDGE_LM, REFLECTION_LM, STATE, TASK_LM, SupportInbox, cost,
                    inbox_feedback, inbox_metric, load_emails, score_inbox)

dspy.configure(lm=TASK_LM)

trainset, valset, testset = load_emails("train"), load_emails("val"), load_emails("test")
program = SupportInbox()
before = {name: pred.signature.instructions for name, pred in program.named_predictors()}

optimizer = dspy.GEPA(
    metric=inbox_feedback,
    auto="light",
    reflection_lm=REFLECTION_LM,
    num_threads=6,
    track_stats=True,
)
optimized = optimizer.compile(program, trainset=trainset, valset=valset)

print("--- instructions, before and after, per predictor ---")
for name, pred in optimized.named_predictors():
    print(f"\n[{name}] before:\n{before[name]}")
    print(f"[{name}] after:\n{pred.signature.instructions}")

print("\n--- search statistics ---")
stats = optimized.detailed_results
print(f"candidates tried   : {len(stats.candidates)}")
print(f"metric calls       : {stats.total_metric_calls}")
print(f"val score by cand. : {[round(s, 3) for s in stats.val_aggregate_scores]}")
print(f"best candidate     : #{stats.best_idx}")

print("\n--- test split, before and after ---")
evaluator = dspy.Evaluate(devset=testset, metric=inbox_metric, num_threads=3)
base_result, gepa_result = evaluator(SupportInbox()), evaluator(optimized)
print(f"baseline : {base_result.score}")
print(f"gepa     : {gepa_result.score}")

# Keep every test reply, before and after, with the judge's verdicts, for the
# hand check in §6 (judge-check.py).
rows = []
for (ex, base_pred, _), (_, gepa_pred, _) in zip(base_result.results, gepa_result.results):
    rows.append({
        "email": ex.email, "gold": f"{ex.category}/{ex.urgency}",
        "before": {"labels": f"{base_pred.category}/{base_pred.urgency}",
                   "reply": base_pred.reply, "judge": score_inbox(ex, base_pred)},
        "after": {"labels": f"{gepa_pred.category}/{gepa_pred.urgency}",
                  "reply": gepa_pred.reply, "judge": score_inbox(ex, gepa_pred)},
    })
STATE.mkdir(exist_ok=True)
(STATE / "inbox-test-replies.json").write_text(json.dumps(rows, indent=2))

print("\n--- one reply, before and after ---")
row = next(r for r in rows if "CSV export" in r["email"])
print(f"email  : {row['email']}")
print(f"before : {row['before']['reply']}")
print(f"after  : {row['after']['reply']}")

optimized.save(STATE / "inbox-gepa.json")
print(f"\nsaved to {STATE.name}/inbox-gepa.json and inbox-test-replies.json")
print(f"cost     : task ${cost(TASK_LM):.3f}  judge ${cost(JUDGE_LM):.3f}  "
      f"reflection ${cost(REFLECTION_LM):.3f}")
