"""Shared pieces for Session 8: the running programs, their metrics, and the
knowledge base the inbox programs retrieve from.

The signatures, modules, metrics, and model roles are Session 7's
`common.py`, unchanged. Session 8 adds the knowledge base loader, the
embedder and retriever, and two new inbox programs:

  GroundedInbox   fixed retrieval: every email searches the knowledge base
                  with the same k, and the draft step sees a `context` input
  AgenticInbox    model-decided retrieval: a ReAct loop owns search as a
                  tool and decides whether and what to retrieve

Both keep SupportInbox's outputs, so Session 7's `inbox_metric` scores them
unchanged. The data loaders read Session 7's JSON by path so the splits
cannot drift.

Model roles, all through OpenRouter:
  TASK_LM        gpt-4.1-nano   the program under test (Sessions 5-7)
  JUDGE_LM       gpt-4.1-mini   the metric's judge (Session 6)
  EMBEDDER       text-embedding-3-small   embeds the knowledge base and queries
"""

import json
import pathlib
import shlex
from typing import Literal

import dspy

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
STATE = HERE / "state"
SESSION7_DATA = HERE.parent.parent / "session-7" / "code" / "data"

TASK_LM = dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=400)
JUDGE_LM = dspy.LM("openrouter/openai/gpt-4.1-mini", max_tokens=600)
EMBEDDER = dspy.Embedder("openrouter/openai/text-embedding-3-small")


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

    async def aforward(self, request: str) -> dspy.Prediction:
        # acall() dispatches here and does not fall back to forward(). A module
        # without aforward cannot run on the event loop; dspy.asyncify wraps
        # such a module in a worker thread instead.
        return await self.predict.acall(request=request)


def load_unix(split: str | None = None) -> list[dspy.Example]:
    rows = json.loads((SESSION7_DATA / "unix-examples.json").read_text())["examples"]
    return [
        dspy.Example(request=r["request"], command=r["command"]).with_inputs("request")
        for r in rows if split is None or r["split"] == split
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
    rows = json.loads((SESSION7_DATA / "support-emails.json").read_text())["examples"]
    return [
        dspy.Example(email=r["email"], category=r["category"], urgency=r["urgency"])
        .with_inputs("email")
        for r in rows if r["split"] == split
    ]


# --- the knowledge base and the retriever (Session 8, §3) -----------------

def load_kb() -> list[str]:
    """The passages, in file order. Each row has an id, a topic, and text."""
    return [row["text"] for row in json.loads((DATA / "kb.json").read_text())["passages"]]


_retriever = None


def retriever(k: int = 3) -> dspy.retrievers.Embeddings:
    """One retriever per process. Embedding the corpus is one API call, cached by litellm."""
    global _retriever
    if _retriever is None:
        _retriever = dspy.retrievers.Embeddings(corpus=load_kb(), embedder=EMBEDDER, k=k)
    return _retriever


def search_knowledge_base(query: str) -> list[str]:
    """Search Planbook's help center and policy pages. Returns the three
    passages most relevant to the query. Use it to find the refund rule, the
    known issue, or the support commitment that applies to a customer's email."""
    return list(retriever()(query).passages)


# --- program 3: GroundedInbox, fixed retrieval -----------------------------

class GroundedDraftReply(dspy.Signature):
    """Write a short, courteous reply to a support email. Use the context
    passages for any policy, timeline, or known issue you mention, and do not
    promise anything the context does not support."""

    email: str = dspy.InputField()
    category: str = dspy.InputField()
    context: list[str] = dspy.InputField(desc="passages from the help center")
    reply: str = dspy.OutputField(desc="2-3 sentences")


class GroundedInbox(dspy.Module):
    """SupportInbox plus one reach: every email searches the knowledge base."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.triage = dspy.Predict(TriageEmail)
        self.extract = dspy.Predict(ExtractTicket)
        self.draft = dspy.ChainOfThought(GroundedDraftReply)
        self.search = retriever(k)   # a plain attribute, not a predictor

    def forward(self, email):
        t = self.triage(email=email)
        ticket = None
        if t.category == "bug_report":
            ticket = self.extract(email=email)
        context = self.search(email).passages
        d = self.draft(email=email, category=t.category, context=context)
        reply = d.reply
        if t.urgency == "high":
            reply = "[PRIORITY] " + reply
        return dspy.Prediction(
            category=t.category, urgency=t.urgency, reply=reply, ticket=ticket,
            context=context,
        )

    async def aforward(self, email):
        t = await self.triage.acall(email=email)
        ticket = None
        if t.category == "bug_report":
            ticket = await self.extract.acall(email=email)
        context = self.search(email).passages   # local numpy; no await needed
        d = await self.draft.acall(email=email, category=t.category, context=context)
        reply = d.reply
        if t.urgency == "high":
            reply = "[PRIORITY] " + reply
        return dspy.Prediction(
            category=t.category, urgency=t.urgency, reply=reply, ticket=ticket,
            context=context,
        )


# --- program 4: AgenticInbox, model-decided retrieval ----------------------

class AgenticInbox(dspy.Module):
    """SupportInbox where the draft step is a ReAct loop that owns search."""

    def __init__(self, max_iters: int = 4):
        super().__init__()
        self.triage = dspy.Predict(TriageEmail)
        self.extract = dspy.Predict(ExtractTicket)
        self.draft = dspy.ReAct(DraftReply, tools=[search_knowledge_base], max_iters=max_iters)

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
            category=t.category, urgency=t.urgency, reply=reply, ticket=ticket,
            trajectory=d.trajectory,
        )


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


# --- shared reporting for the inbox programs --------------------------------

def evaluate_inbox(program, testset, num_threads: int = 3):
    """Score on the test split and break the score into the four checks.

    Returns (score, per_check_totals, rows). Re-scoring each prediction with
    score_inbox hits the judge cache, so the breakdown costs nothing extra.
    """
    evaluator = dspy.Evaluate(devset=testset, metric=inbox_metric, num_threads=num_threads)
    result = evaluator(program)
    totals = {"category": 0, "urgency": 0, "grounded": 0, "actionable": 0}
    rows = []
    for example, prediction, score in result.results:
        checks = score_inbox(example, prediction)
        for name, ok in checks.items():
            totals[name] += int(ok)
        rows.append((example, prediction, score, checks))
    return result.score, totals, rows


def print_breakdown(label: str, score: float, totals: dict, n: int):
    checks = "  ".join(f"{name} {count}/{n}" for name, count in totals.items())
    print(f"{label:10s} score {score:>5.1f}   {checks}")
