"""The reward function (§3): score one reply with no gold label.

A metric receives the gold example, so it can check the labels. A reward
function receives only the program's inputs and its prediction. It runs
while the program works, when no gold label exists.

This script runs Session 8's grounded inbox on the six test emails and
scores every reply with `reply_reward`. Nothing is retried yet. The point
is to see what the reward says about the replies we already have.
"""

import dspy

from common import (TASK_LM, GroundedInbox, load_emails, judge_reply, redirects_customer,
                    reply_reward, spent)

dspy.configure(lm=TASK_LM)

print("--- the hard rule: one line of Python ---")
for reply in ["Please email support@planbook.example with the charge dates.",
              "We have your report and will answer within two business days."]:
    print(f"redirects_customer = {redirects_customer(reply)!s:5s} | {reply}")

def mark(ok: bool) -> str:
    return "pass" if ok else "FAIL"


print("\n--- one reply through the reward function ---")
inbox = GroundedInbox()
email = "I was charged twice for my March invoice. Please refund the duplicate."
pred = inbox(email=email)
print(f"email  : {email}")
print(f"reply  : {pred.reply}")
print(f"reward : {reply_reward({'email': email}, pred):.2f}")

print("\n--- the six test replies from Session 8, scored by the reward ---")
print(f"{'hard rule':>9s} {'grounded':>9s} {'actionable':>11s} {'reward':>7s}   email")
rewards = []
for example in load_emails("test"):
    pred = inbox(email=example.email)
    reward = reply_reward({"email": example.email}, pred)
    rewards.append(reward)
    if redirects_customer(pred.reply):       # the judge is never called
        columns = f"{'FAIL':>9s} {'-':>9s} {'-':>11s}"
    else:
        checks = judge_reply(example.email, pred.reply)
        columns = f"{'pass':>9s} {mark(checks['grounded']):>9s} {mark(checks['actionable']):>11s}"
    print(f"{columns} {reward:>7.2f}   {example.email[:42]}")

passing = sum(r >= 1.0 for r in rewards)
print(f"\nreplies at reward 1.0 : {passing} of {len(rewards)}")
print(f"mean reward           : {sum(rewards) / len(rewards):.2f}")
print(f"cost : ${spent():.4f}")

assert passing < len(rewards), "the reward should find something to fix"
assert reply_reward({"email": "x"}, dspy.Prediction(reply="Write to support@planbook.example")) < 1.0
