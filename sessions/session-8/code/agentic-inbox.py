"""Model-decided retrieval (§5): the agentic inbox.

§3 searched on every email with a fixed k. Here the draft step is a ReAct
loop with search_knowledge_base as its one tool. The model decides whether
to search, what to search for, and when it has enough. Same signature, same
outputs, same metric, same six test emails.

The table at the end answers one question: does letting the model decide
beat always searching, and what does it cost per email.
"""

import json

import dspy

from common import (JUDGE_LM, STATE, TASK_LM, AgenticInbox, GroundedInbox, SupportInbox,
                    cost, evaluate_inbox, load_emails, print_breakdown)

dspy.configure(lm=TASK_LM)

agentic = AgenticInbox(max_iters=4)

print("--- the predictor tree ---")
print([name for name, _ in agentic.named_predictors()])

print("\n--- one bug email through the loop ---")
email = "URGENT: the export button crashes the app for our whole team, we ship today!"
out = agentic(email=email)
for key, value in out.trajectory.items():
    text = str(value).replace("\n", " ")
    print(f"{key:14s}: {text[:110]}{'...' if len(text) > 110 else ''}")
print(f"\nreply : {out.reply}")

print("\n--- one feedback email: does it search at all? ---")
praise = "Just wanted to say the new dashboard layout is a big improvement. Nice work."
out2 = agentic(email=praise)
print(f"tool calls : {[v for k, v in out2.trajectory.items() if k.startswith('tool_name')]}")
print(f"reply      : {out2.reply}")

print("\n--- test split: three programs, same metric ---")
testset = load_emails("test")
n = len(testset)
table = []
for label, program in [("baseline", SupportInbox()), ("grounded", GroundedInbox(k=3)),
                       ("agentic", agentic)]:
    calls_before = len(TASK_LM.history)
    score, totals, rows = evaluate_inbox(program, testset)
    calls = (len(TASK_LM.history) - calls_before) / n
    searches = sum(
        sum(1 for k, v in pred.trajectory.items() if k.startswith("tool_name") and v != "finish")
        for _, pred, _, _ in rows
    ) / n if label == "agentic" else (1.0 if label == "grounded" else 0.0)
    print_breakdown(label, score, totals, n)
    table.append((label, score, calls, searches, rows))

print(f"\n{'program':10s} {'score':>6s} {'LM calls/email':>15s} {'searches/email':>15s}")
for label, score, calls, searches, _ in table:
    print(f"{label:10s} {score:>6.1f} {calls:>15.1f} {searches:>15.1f}")
print("(LM calls counted from the task model's history; the judge's calls are not included)")

rows = []
for (ex, _, _, _), (_, agent_pred, _, agent_checks) in zip(table[0][4], table[2][4]):
    searches = [v for k, v in agent_pred.trajectory.items() if k.startswith("tool_args")]
    rows.append({"email": ex.email, "gold": f"{ex.category}/{ex.urgency}",
                 "searches": searches, "reply": agent_pred.reply, "checks": agent_checks})
STATE.mkdir(exist_ok=True)
(STATE / "agentic-test-replies.json").write_text(json.dumps(rows, indent=2))
print(f"\nsaved replies and search queries to {STATE.name}/agentic-test-replies.json")
print(f"cost : task ${cost(TASK_LM):.4f}  judge ${cost(JUDGE_LM):.4f}")
