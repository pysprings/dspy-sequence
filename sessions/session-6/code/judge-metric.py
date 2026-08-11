"""A composite LLM-judge metric (§6): scoring Session 5's SupportInbox.

Session 5 ended on a question: the model rated the billing email `high`
urgency; was that *correct*? Rules can check the label against gold. Rules
cannot check whether the drafted reply is any good. So the metric mixes both:
cheap assertions where rules work, an LLM judge where they don't, folded
into one score.

The judge is itself a DSPy program (a Signature run by Predict), which means
it is inspectable with inspect_history and, in Session 7, optimizable.
"""

from typing import Literal

import dspy

dspy.configure(lm=dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=400))
judge_lm = dspy.LM("openrouter/openai/gpt-4.1-mini", max_tokens=400)


# --- the program under test: Session 5's SupportInbox, unchanged -----------

class TriageEmail(dspy.Signature):
    """Classify an incoming support email."""

    email: str = dspy.InputField()
    category: Literal["billing", "bug_report", "feedback"] = dspy.OutputField()
    urgency: Literal["low", "high"] = dspy.OutputField()


class ExtractTicket(dspy.Signature):
    """Pull the fields an engineering ticket needs from a bug report."""

    email: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="one-line bug summary")
    affected_component: str = dspy.OutputField()


class DraftReply(dspy.Signature):
    """Write a short, courteous reply to a support email."""

    email: str = dspy.InputField()
    category: str = dspy.InputField()
    reply: str = dspy.OutputField(desc="2-3 sentences")


class SupportInbox(dspy.Module):
    def __init__(self):
        super().__init__()
        self.triage = dspy.Predict(TriageEmail)
        self.extract = dspy.Predict(ExtractTicket)
        self.draft = dspy.ChainOfThought(DraftReply)

    def forward(self, email):
        t = self.triage(email=email)
        ticket = None
        if t.category == "bug_report":
            ticket = self.extract(email=email)
        d = self.draft(email=email, category=t.category)
        reply = d.reply
        if t.urgency == "high":
            reply = "[PRIORITY] " + reply
        return dspy.Prediction(
            category=t.category, urgency=t.urgency, reply=reply, ticket=ticket
        )


# --- gold labels: Session 2's craft, applied ------------------------------
# Urgency now has an answer key. The billing double-charge is labeled high
# on purpose: money already lost. The feature request is the low-urgency foil.

devset = [
    dspy.Example(
        email="I was charged twice for my March invoice. Please refund the duplicate.",
        category="billing", urgency="high",
    ).with_inputs("email"),
    dspy.Example(
        email="URGENT: the export button crashes the app for our whole team, we ship today!",
        category="bug_report", urgency="high",
    ).with_inputs("email"),
    dspy.Example(
        email="Just wanted to say the new dashboard layout is a big improvement. Nice work.",
        category="feedback", urgency="low",
    ).with_inputs("email"),
    dspy.Example(
        email="Could you add a CSV export option someday? Not blocking, just handy.",
        category="feedback", urgency="low",
    ).with_inputs("email"),
    dspy.Example(
        email="My card was declined but you still marked the invoice unpaid twice?",
        category="billing", urgency="high",
    ).with_inputs("email"),
    dspy.Example(
        email="Minor thing: the settings page misspells 'preferences' in the sidebar.",
        category="bug_report", urgency="low",
    ).with_inputs("email"),
]


# --- the judge: a metric that is itself a DSPy program --------------------

class AssessReply(dspy.Signature):
    """Assess a support reply along one stated dimension."""

    email: str = dspy.InputField(desc="the customer's email")
    reply: str = dspy.InputField(desc="the support reply under assessment")
    question: str = dspy.InputField(desc="the yes/no dimension to assess")
    verdict: bool = dspy.OutputField()


assess = dspy.Predict(AssessReply)


def inbox_metric(example, pred, trace=None):
    """Two rule checks plus two judged checks, equally weighted."""
    # Rules where rules work: gold labels exist for these.
    category_ok = pred.category == example.category
    urgency_ok = pred.urgency == example.urgency

    # A judge where they don't: no gold reply exists, only properties.
    with dspy.context(lm=judge_lm):
        grounded = assess(
            email=example.email, reply=pred.reply,
            question="Does the reply address this customer's specific "
                     "situation rather than being a generic acknowledgement?",
        ).verdict
        actionable = assess(
            email=example.email, reply=pred.reply,
            question="Does the reply state a concrete next step or "
                     "resolution the customer can expect?",
        ).verdict

    checks = [category_ok, urgency_ok, grounded, actionable]
    if trace is not None:              # bootstrapping: demos must be clean
        return all(checks)
    return sum(checks) / len(checks)   # evaluation: partial credit


inbox = SupportInbox()
evaluator = dspy.Evaluate(devset=devset, metric=inbox_metric, num_threads=3)
result = evaluator(inbox)

print(f"\noverall: {result.score}")
print("\n--- per-example breakdown ---")
for example, prediction, score in result.results:
    print(f"{score:>5}  {example.category:>10}/{example.urgency:<4}  "
          f"got {prediction.category}/{prediction.urgency}  "
          f"| {example.email[:48]}...")

# Uncomment to read the judge's last conversation, like any other predictor:
# dspy.inspect_history(n=1)
