"""A real program (§6): multiple predictors plus plain Python control flow.

Triage first, then branch in ordinary Python to decide which predictors
run next: the "working AI program" of the session title.
"""

from typing import Literal

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=400))


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
        # forward is just Python: any control flow or non-LM code is legal
        # here, and every sub-call below is individually auditable.
        t = self.triage(email=email)
        ticket = None
        if t.category == "bug_report":
            # Only bug reports pay for the extraction call -- plain branching.
            ticket = self.extract(email=email)
        d = self.draft(email=email, category=t.category)
        reply = d.reply
        if t.urgency == "high":
            reply = "[PRIORITY] " + reply
        return dspy.Prediction(
            category=t.category, urgency=t.urgency, reply=reply, ticket=ticket
        )


inbox = SupportInbox()

print("--- the program's predictor tree ---")
# The CoT predictor shows up as "draft.predict" -- it lives one level down.
for name, _ in inbox.named_predictors():
    print(f"  {name}")

emails = [
    "I was charged twice for my March invoice. Please refund the duplicate.",
    "URGENT: the export button crashes the app for our whole team, we ship today!",
    "Just wanted to say the new dashboard layout is a big improvement. Nice work.",
]

for email in emails:
    out = inbox(email=email)
    print(f"\n--- {email[:50]}... ---")
    print(f"category : {out.category}")
    print(f"urgency  : {out.urgency}")
    print(f"reply    : {out.reply[:100]}")
    assert out.category in ("billing", "bug_report", "feedback"), out.category

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
