"""Shared pieces for Session 7: the two programs, their data, and their metrics.

Sessions 5 and 6 repeated the program and metric in every script. Session 7
runs each program through several optimizers, so the repeated parts live here.
Nothing in this file is new: the signatures, modules, and three-tier / judge
metrics are Session 5 and 6 code, unchanged. The only additions are the
data loaders and the five-argument metric variants GEPA requires (§5, §6).

Model roles, all through OpenRouter:
  TASK_LM        gpt-4.1-nano   the program under optimization (Sessions 5-6)
  JUDGE_LM       gpt-4.1-mini   the metric's judge (Session 6)
  REFLECTION_LM  gpt-5.6-sol    reads traces and proposes instructions (GEPA)
  DEPLOY_LM      gpt-5.6-luna   a different task model, for the transfer test (§7)
"""

import json
import pathlib
import shlex
from typing import Literal

import dspy

DATA = pathlib.Path(__file__).parent / "data"
STATE = pathlib.Path(__file__).parent / "state"

TASK_LM = dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=400)
JUDGE_LM = dspy.LM("openrouter/openai/gpt-4.1-mini", max_tokens=600)
REFLECTION_LM = dspy.LM("openrouter/openai/gpt-5.6-sol", max_tokens=16000, temperature=1.0)
DEPLOY_LM = dspy.LM("openrouter/openai/gpt-5.6-luna", max_tokens=400)


def cost(*lms) -> float:
    """Dollars spent so far across the given LMs' call histories (cache hits are free)."""
    return sum((entry.get("cost") or 0.0) for lm in lms for entry in lm.history)


# --- program 1: EnglishToUnix, wrapped as in Session 5 --------------------

class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


class UnixAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(EnglishToUnix)

    def forward(self, request: str) -> dspy.Prediction:
        return self.predict(request=request)


def load_unix(split: str) -> list[dspy.Example]:
    rows = json.loads((DATA / "unix-examples.json").read_text())["examples"]
    return [
        dspy.Example(request=r["request"], command=r["command"]).with_inputs("request")
        for r in rows if r["split"] == split
    ]


# --- program 2: SupportInbox, Session 5 code unchanged --------------------

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


def load_emails(split: str) -> list[dspy.Example]:
    rows = json.loads((DATA / "support-emails.json").read_text())["examples"]
    return [
        dspy.Example(email=r["email"], category=r["category"], urgency=r["urgency"])
        .with_inputs("email")
        for r in rows if r["split"] == split
    ]


# --- metric 1: the three-tier command metric (Session 6 §5) ---------------

class CommandsEquivalent(dspy.Signature):
    """Decide whether two Unix commands are functionally equivalent for the
    user's request: would they produce the same effect and output?"""

    request = dspy.InputField(desc="the user's natural language request")
    expected = dspy.InputField(desc="the known-good command")
    predicted = dspy.InputField(desc="the command under evaluation")
    equivalent: bool = dspy.OutputField()


judge = dspy.ChainOfThought(CommandsEquivalent)


def exact_match(expected: str, predicted: str) -> bool:
    return expected.strip() == predicted.strip()


def normalized_match(expected: str, predicted: str) -> bool:
    try:
        return shlex.split(expected.strip()) == shlex.split(predicted.strip())
    except ValueError:  # unbalanced quotes etc.: fall back to tier 1
        return exact_match(expected, predicted)


def score_command(example, pred) -> tuple[float, str]:
    """The three tiers, plus a sentence saying which tier decided."""
    expected, predicted = example.command, pred.command
    if exact_match(expected, predicted):
        return 1.0, "exact match"
    if normalized_match(expected, predicted):
        return 0.8, f"matches after normalizing quotes/whitespace; gold was {expected!r}"
    with dspy.context(lm=JUDGE_LM):
        verdict = judge(request=example.request, expected=expected, predicted=predicted)
    if verdict.equivalent:
        return 1.0, f"judge: equivalent to gold {expected!r}"
    return 0.0, f"not equivalent: gold is {expected!r}, predicted {predicted!r}"


def command_match(example, pred, trace=None):
    """Session 6's metric: graded when scoring, strict when tracing."""
    score, _ = score_command(example, pred)
    if trace is not None:          # bootstrapping: pass/fail only
        return score >= 0.8
    return score                   # evaluation: partial credit is signal


def command_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """The same tiers in GEPA's shape: five arguments in, score plus feedback out."""
    score, why = score_command(gold, pred)
    return dspy.Prediction(score=score, feedback=why)


# --- metric 2: the composite judge metric (Session 6 §6) ------------------

class AssessReply(dspy.Signature):
    """Assess a support reply along one stated dimension."""

    email: str = dspy.InputField(desc="the customer's email")
    reply: str = dspy.InputField(desc="the support reply under assessment")
    question: str = dspy.InputField(desc="the yes/no dimension to assess")
    verdict: bool = dspy.OutputField()


assess = dspy.Predict(AssessReply)

GROUNDED_Q = ("Does the reply address this customer's specific situation rather "
              "than being a generic acknowledgement?")
ACTIONABLE_Q = ("Does the reply state a concrete next step or resolution the "
                "customer can expect?")


def score_inbox(example, pred) -> dict[str, bool]:
    """Two rule checks plus two judged checks, as named booleans."""
    with dspy.context(lm=JUDGE_LM):
        grounded = assess(email=example.email, reply=pred.reply, question=GROUNDED_Q).verdict
        actionable = assess(email=example.email, reply=pred.reply, question=ACTIONABLE_Q).verdict
    return {
        "category": pred.category == example.category,
        "urgency": pred.urgency == example.urgency,
        "grounded": grounded,
        "actionable": actionable,
    }


def inbox_metric(example, pred, trace=None):
    """Session 6's metric: equal-weighted checks, strict when tracing."""
    checks = list(score_inbox(example, pred).values())
    if trace is not None:              # bootstrapping: demos must be clean
        return all(checks)
    return sum(checks) / len(checks)   # evaluation: partial credit


def inbox_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA's shape. Same score for every predictor; feedback routed by pred_name."""
    checks = score_inbox(gold, pred)
    score = sum(checks.values()) / len(checks)

    label_notes = []
    if not checks["category"]:
        label_notes.append(f"category should be {gold.category!r}, got {pred.category!r}")
    if not checks["urgency"]:
        label_notes.append(f"urgency should be {gold.urgency!r}, got {pred.urgency!r}")
    reply_notes = []
    if not checks["grounded"]:
        reply_notes.append("the reply reads as a generic acknowledgement; it should "
                           "name the customer's specific situation")
    if not checks["actionable"]:
        reply_notes.append("the reply promises nothing concrete; it should state a "
                           "next step or resolution the customer can expect")

    if pred_name == "triage":
        notes = label_notes or ["labels correct"]
    elif pred_name == "draft.predict":
        notes = reply_notes or ["reply passed both judge checks"]
    elif pred_name == "extract":
        notes = ["ticket extraction is not scored by this metric"]
    else:
        notes = (label_notes + reply_notes) or ["all four checks passed"]
    return dspy.Prediction(score=score, feedback="; ".join(notes))
