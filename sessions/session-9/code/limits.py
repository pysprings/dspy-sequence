"""What the retry loops do not do (§7).

Three short demonstrations:

  1. A loop that never reaches the threshold still returns a reply, and the
     reply carries no sign that it fell short. The caller must check.
  2. An attempt that raises an exception counts as an attempt. If every
     attempt raises, the loop raises the last exception.
  3. The wrapped predictors stay visible to optimizers and to save().
     Refine's hint writer does not.
"""

import dspy

from common import HINT_LM, TASK_LM, load_emails, reply_reward, spent, with_retry, wrapped

dspy.configure(lm=TASK_LM)
testset = load_emails("test")

# --- below the threshold --------------------------------------------------

print("--- the loop returns a reply either way ---")
inbox, recorder = with_retry(dspy.Refine, N=3, threshold=1.0)
praise = testset[5].email
with dspy.context(lm=HINT_LM):             # the stronger hint writer from §5
    pred = inbox(email=praise)
print(wrapped("email         : ", praise))
print(f"rewards tried : {[reward for reward, _ in recorder.attempts[praise]]}")
print(wrapped("returned      : ", pred.reply))
print(f"fields        : {sorted(pred.keys())}")


def send_or_hold(email: str, pred, threshold: float = 1.0) -> str:
    """Score the reply we are about to send. Below the threshold, a person reads it first."""
    return "send" if reply_reward({"email": email}, pred) >= threshold else "hold"


print("\n--- the same six emails, with a check before sending ---")
decisions = []
with dspy.context(lm=HINT_LM):
    for example in testset:
        pred = inbox(email=example.email)
        decision = send_or_hold(example.email, pred)
        decisions.append(decision)
        print(f"{decision:5s} | {example.email[:66]}")
print(f"sent {decisions.count('send')} of {len(decisions)}, held {decisions.count('hold')} for a person")

# --- exceptions -------------------------------------------------------------

print("\n--- an attempt that raises ---")


class Flaky(dspy.Module):
    """Fails a set number of times, then answers. Stands in for a timeout or a parse error."""

    remaining_failures = 0

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        if Flaky.remaining_failures > 0:
            Flaky.remaining_failures -= 1
            raise TimeoutError("the model did not answer")
        return self.predict(question=question)


def any_answer(args, pred) -> float:
    return 1.0


for failures in (1, 3):
    Flaky.remaining_failures = failures
    loop = dspy.BestOfN(module=Flaky(), N=3, reward_fn=any_answer, threshold=1.0)
    try:
        answer = loop(question="What is 2 + 2?").answer
        print(f"{failures} failure(s) in 3 attempts -> answer: {answer}")
    except TimeoutError as error:
        print(f"{failures} failure(s) in 3 attempts -> raised {type(error).__name__}: {error}")

# --- what an optimizer and save() can see ---------------------------------------

print("\n--- the predictor tree of the Refine program ---")
print([name for name, _ in inbox.named_predictors()])

saved = inbox.dump_state()                 # what save() writes to JSON
print(f"keys in the saved state : {list(saved)}")
print(f"cost : ${spent():.4f}")

assert "reward" not in pred.keys()
assert "hold" in decisions and "send" in decisions
assert list(saved) == ["triage", "extract", "draft.module.predict"]
