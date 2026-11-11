"""Compile time or run time (§8): put the lesson in the prompt.

In §5 the hint writer told the draft step the same thing on three emails:
the customer is already writing to support, so do not send them there.
A lesson that every email needs does not belong in a retry loop. It
belongs in the instructions, where one draft can use it.

This script adds that sentence to the draft signature by hand and runs the
plain program, with no loop, three times. Then it puts Refine around the
edited program, to see what the loop still adds.
"""

import dspy

from common import (HINT_LM, TASK_LM, GroundedDraftReply, GroundedInbox, load_emails, run_test,
                    spent, with_retry, wrapped)

dspy.configure(lm=TASK_LM)
testset = load_emails("test")

LESSON = (" The customer is already writing to support. Do not ask them to email"
          " support. Say what support will do next.")
EditedDraftReply = GroundedDraftReply.with_instructions(GroundedDraftReply.instructions + LESSON)

print("--- the edited instructions ---")
print(GroundedDraftReply.instructions)
print(wrapped("ADDED: ", LESSON.strip()))


def edited(program: GroundedInbox) -> GroundedInbox:
    """Swap the edited signature into the draft predictor, inside a loop or not."""
    for name, predictor in program.named_predictors():
        if name.startswith("draft"):
            predictor.signature = predictor.signature.with_instructions(EditedDraftReply.instructions)
    return program


rows = {"one draft, edited prompt": [], "Refine, edited prompt": []}
for start in (0, 100, 200):
    task_lm = TASK_LM.copy(rollout_id=start) if start else TASK_LM
    plain = edited(GroundedInbox())
    plain.set_lm(task_lm)
    rows["one draft, edited prompt"].append(run_test(plain, testset))
    looped, recorder = with_retry(dspy.Refine, N=3, threshold=1.0, task_lm=task_lm)
    with dspy.context(lm=HINT_LM):
        rows["Refine, edited prompt"].append(run_test(edited(looped), testset, recorder))

print("\n--- test split, three runs ---")
print(f"{'program':26s} {'run 1':>6s} {'run 2':>6s} {'run 3':>6s} {'mean':>6s} {'redirects':>10s} {'attempts':>9s} {'hints':>6s}")
for label, runs in rows.items():
    scores = [r["score"] for r in runs]
    redirects = sum(r["redirects"] for r in runs)
    drafts = sum(r["attempts"] for r in runs) / 3
    hints = sum(r["hints"] for r in runs) / 3
    print(f"{label:26s} {scores[0]:>6.1f} {scores[1]:>6.1f} {scores[2]:>6.1f} {sum(scores) / 3:>6.1f} "
          f"{redirects:>7d}/18 {drafts:>9.1f} {hints:>6.1f}")
print(f"cost : ${spent():.4f}")

assert sum(r["redirects"] for r in rows["one draft, edited prompt"]) < 8, "§6 measured 8 of 18 without the edit"
