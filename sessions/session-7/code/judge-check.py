"""Judge check (§6): did GEPA improve the replies, or learn to please the judge?

Session 6 said a judge must be re-validated once an optimizer enters.
This script compares the judge's verdicts on the six test replies, before
and after GEPA, against hand marks in state/inbox-hand-marks.json. The
marks use the same two questions the judge answers. Agreement is plain
percent over 24 verdicts (6 emails x 2 dimensions x before/after).
"""

import json

from common import STATE

replies = json.loads((STATE / "inbox-test-replies.json").read_text())
marks = json.loads((STATE / "inbox-hand-marks.json").read_text())

print("--- rubric ---")
print(f"grounded   : {marks['rubric']['grounded']}")
print(f"actionable : {marks['rubric']['actionable']}")

for marker, hand in marks["marks"].items():
    if not hand:
        print(f"\n{marker}: no marks yet")
        continue
    agree = total = 0
    print(f"\n--- {marker} vs judge ---")
    print(f"{'email':30s} {'stage':6s} {'dim':10s} {'judge':5s} {'hand':5s}")
    for row, hand_row in zip(replies, hand):
        for stage in ("before", "after"):
            for dim in ("grounded", "actionable"):
                j, h = row[stage]["judge"][dim], hand_row[stage][dim]
                agree += j == h
                total += 1
                flag = "" if j == h else "  <- disagree"
                print(f"{row['email'][:30]:30s} {stage:6s} {dim:10s} {str(j):5s} {str(h):5s}{flag}")
    print(f"agreement: {agree}/{total} = {100 * agree / total:.0f}%")

print("\n--- reply quality by hand, before -> after (sum of two dimensions, max 12) ---")
for marker, hand in marks["marks"].items():
    if hand:
        before = sum(r["before"]["grounded"] + r["before"]["actionable"] for r in hand)
        after = sum(r["after"]["grounded"] + r["after"]["actionable"] for r in hand)
        print(f"{marker:8s}: {before} -> {after}")
judge_before = sum(r["before"]["judge"]["grounded"] + r["before"]["judge"]["actionable"] for r in replies)
judge_after = sum(r["after"]["judge"]["grounded"] + r["after"]["judge"]["actionable"] for r in replies)
print(f"{'judge':8s}: {judge_before} -> {judge_after}")
