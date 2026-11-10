"""Fixed retrieval (§3): the grounded inbox.

Session 7 ended with replies that were polite and promised nothing, and a
note that an optimizer cannot add a step the program does not have. This
script adds the step: every email searches the knowledge base, and the draft
predictor sees the passages as a `context` input. The retriever is a plain
attribute, not a predictor, so the predictor tree does not change.

The score is Session 7's inbox metric on the same six test emails, broken
into its four checks. The claim under test is that `actionable` rises.
"""

import json

import dspy

from common import (JUDGE_LM, STATE, TASK_LM, GroundedInbox, SupportInbox, cost,
                    evaluate_inbox, load_emails, load_kb, print_breakdown)

dspy.configure(lm=TASK_LM)

print("--- the knowledge base ---")
passages = load_kb()
print(f"{len(passages)} passages, {sum(len(p) for p in passages)} characters")

grounded = GroundedInbox(k=3)

print("\n--- what retrieval returns for one bug email ---")
email = "URGENT: the export button crashes the app for our whole team, we ship today!"
for i, passage in enumerate(grounded.search(email).passages, 1):
    print(f"[{i}] {passage[:150]}{'...' if len(passage) > 150 else ''}")

print("\n--- the predictor tree, before and after the reach ---")
for label, program in [("SupportInbox", SupportInbox()), ("GroundedInbox", grounded)]:
    print(f"{label:14s}: {[name for name, _ in program.named_predictors()]}")
assert "search" not in dict(grounded.named_predictors())

print("\n--- one email through the grounded inbox ---")
out = grounded(email=email)
print(f"category : {out.category}")
print(f"urgency  : {out.urgency}")
print(f"reply    : {out.reply}")

print("\n--- test split: baseline vs grounded, by check ---")
testset = load_emails("test")
n = len(testset)
base_score, base_totals, base_rows = evaluate_inbox(SupportInbox(), testset)
grd_score, grd_totals, grd_rows = evaluate_inbox(grounded, testset)
print_breakdown("baseline", base_score, base_totals, n)
print_breakdown("grounded", grd_score, grd_totals, n)

print("\n--- the replies, side by side ---")
rows = []
for (ex, base_pred, _, base_checks), (_, grd_pred, _, grd_checks) in zip(base_rows, grd_rows):
    print(f"\n{ex.email[:70]}")
    print(f"  baseline [g={int(base_checks['grounded'])} a={int(base_checks['actionable'])}]: {base_pred.reply[:160]}")
    print(f"  grounded [g={int(grd_checks['grounded'])} a={int(grd_checks['actionable'])}]: {grd_pred.reply[:160]}")
    rows.append({
        "email": ex.email, "gold": f"{ex.category}/{ex.urgency}",
        "baseline": {"reply": base_pred.reply, "checks": base_checks},
        "grounded": {"reply": grd_pred.reply, "checks": grd_checks,
                     "context": list(grd_pred.context)},
    })

STATE.mkdir(exist_ok=True)
(STATE / "grounded-test-replies.json").write_text(json.dumps(rows, indent=2))
print(f"\nsaved replies and retrieved context to {STATE.name}/grounded-test-replies.json")
print(f"cost : task ${cost(TASK_LM):.4f}  judge ${cost(JUDGE_LM):.4f}")
