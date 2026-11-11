"""Shared pieces for Session 9: the grounded inbox, its metric, and the reward
function that the retry loops score with.

The signatures, the GroundedInbox program, the judge, and `inbox_metric` are
Session 8's `common.py`, unchanged. The loaders read Session 7's emails and
Session 8's knowledge base by path, so the data cannot drift.

Session 9 adds:

  reply_reward    the reward function: one hard rule, then two judged checks
  Recorder        wraps a reward function and keeps every attempt it scored
  with_retry      puts BestOfN or Refine around the draft step
  run_test        scores one program on the test split and counts the calls

Model roles, all through OpenRouter:
  TASK_LM        gpt-4.1-nano   the program under test
  JUDGE_LM       gpt-4.1-mini   the judge, in the metric and in the reward
  HINT_LM        gpt-4.1-mini   writes Refine's hints in §5 (same model as the judge)
  SECOND_JUDGE   claude-haiku-4.5   re-scores the final replies in §6
  EMBEDDER       text-embedding-3-small   embeds the knowledge base and queries
"""

import json
import logging
import pathlib
import re
import textwrap
from typing import Literal

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY

# Evaluate logs a line per run, and Refine logs three type warnings per hint
# because it passes numbers to its hint writer as text. Neither helps here.
logging.getLogger("dspy").setLevel(logging.ERROR)

HERE = pathlib.Path(__file__).parent
STATE = HERE / "state"
SESSION7_DATA = HERE.parent.parent / "session-7" / "code" / "data"
SESSION8_DATA = HERE.parent.parent / "session-8" / "code" / "data"

TASK_LM = dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=400)
JUDGE_LM = dspy.LM("openrouter/openai/gpt-4.1-mini", max_tokens=600)
HINT_LM = JUDGE_LM
SECOND_JUDGE = dspy.LM("openrouter/anthropic/claude-haiku-4.5", max_tokens=600)
EMBEDDER = dspy.Embedder("openrouter/openai/text-embedding-3-small")


def spent() -> float:
    """Dollars this process's calls cost, all models. A cached call counts at
    its original price, so the figure is what the script costs with no cache.

    The retry loops copy the LM for each attempt, so one LM's own history
    misses most calls. The global history sees all of them.
    """
    return sum((entry.get("cost") or 0.0) for entry in GLOBAL_HISTORY)


# --- the inbox program, Session 8 code unchanged ---------------------------

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


class GroundedDraftReply(dspy.Signature):
    """Write a short, courteous reply to a support email. Use the context
    passages for any policy, timeline, or known issue you mention, and do not
    promise anything the context does not support."""

    email: str = dspy.InputField()
    category: str = dspy.InputField()
    context: list[str] = dspy.InputField(desc="passages from the help center")
    reply: str = dspy.OutputField(desc="2-3 sentences")


def load_emails(split: str) -> list[dspy.Example]:
    rows = json.loads((SESSION7_DATA / "support-emails.json").read_text())["examples"]
    return [
        dspy.Example(email=r["email"], category=r["category"], urgency=r["urgency"])
        .with_inputs("email")
        for r in rows if r["split"] == split
    ]


def load_kb() -> list[str]:
    return [row["text"] for row in json.loads((SESSION8_DATA / "kb.json").read_text())["passages"]]


_retriever = None


def retriever(k: int = 3) -> dspy.retrievers.Embeddings:
    """One retriever per process. Embedding the corpus is one API call, cached by litellm."""
    global _retriever
    if _retriever is None:
        _retriever = dspy.retrievers.Embeddings(corpus=load_kb(), embedder=EMBEDDER, k=k)
    return _retriever


class GroundedInbox(dspy.Module):
    """Triage, extract, search the knowledge base, draft. Session 8, §3."""

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


# --- the metric, Session 6 code unchanged ----------------------------------

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
    if trace is not None:
        return all(checks)
    return sum(checks) / len(checks)


# --- Session 9: the reward function -----------------------------------------

def redirects_customer(reply: str) -> bool:
    """True if the reply tells the customer to email support. They already did."""
    return "support@" in reply.lower()


def judge_reply(email: str, reply: str) -> dict[str, bool]:
    """The metric's two judged checks. Neither needs a gold label."""
    with dspy.context(lm=JUDGE_LM):
        grounded = assess(email=email, reply=reply, question=GROUNDED_Q).verdict
        actionable = assess(email=email, reply=reply, question=ACTIONABLE_Q).verdict
    return {"grounded": grounded, "actionable": actionable}


def reply_reward(args, pred) -> float:
    """Score one draft reply from 0.0 to 1.0.

    A hard rule comes first. If the reply tells the customer to email
    support@..., the reward is 0.0. The customer is already writing to
    support, so the reply must handle the request itself.

    Two judged checks follow, each worth half a point:
    grounded:   the reply addresses this customer's specific situation and
                is not a generic acknowledgement.
    actionable: the reply states a concrete next step or resolution that
                the customer can expect.
    """
    if redirects_customer(pred.reply):
        return 0.0
    checks = judge_reply(args["email"], pred.reply)
    return (checks["grounded"] + checks["actionable"]) / 2


# --- Session 9: the retry loops ----------------------------------------------

class Recorder:
    """A reward function that remembers every attempt it scored.

    BestOfN and Refine return only the winning prediction. To show the
    attempts they threw away, score through a Recorder and read
    `attempts[email]` afterwards: a list of (reward, reply) in the order tried.
    """

    def __init__(self, reward_fn):
        self.reward_fn = reward_fn
        self.attempts: dict[str, list[tuple[float, str]]] = {}

    def __call__(self, args, pred) -> float:
        reward = self.reward_fn(args, pred)
        self.attempts.setdefault(args["email"], []).append((reward, pred.reply))
        return reward


def with_retry(wrapper, N: int = 3, threshold: float = 1.0,   # noqa: N803
               task_lm: dspy.LM = TASK_LM) -> tuple[GroundedInbox, Recorder]:
    """A grounded inbox whose draft step is wrapped in BestOfN or Refine.

    Returns the program and a Recorder that holds every attempt.

    Every predictor is pinned to `task_lm`. The retry loops then draft with
    the task model whatever `dspy.configure` says, and Refine's hint writer,
    which has no pin, uses the configured model.

    The wrapper is built with the plain reward function, because Refine reads
    that function's source code. The recorder is swapped in afterwards.
    """
    inbox = GroundedInbox()
    inbox.draft = wrapper(module=inbox.draft, N=N, reward_fn=reply_reward, threshold=threshold)
    recorder = Recorder(reply_reward)
    inbox.draft.reward_fn = recorder
    inbox.set_lm(task_lm)
    return inbox, recorder


HINT_WRITER_MARK = "assign blame to each module"   # from the hint writer's instructions


def hints_written(since: int = 0) -> list[str]:
    """The text of every hint that Refine's hidden hint writer has produced.

    Refine does not return its hints, so we read them from the global call
    history. `since` skips that many earlier hints.
    """
    hints = []
    for entry in GLOBAL_HISTORY:
        if HINT_WRITER_MARK not in entry["messages"][0]["content"]:
            continue
        raw = entry["outputs"][0]
        raw = raw if isinstance(raw, str) else raw.get("text", "")
        advice = raw.split("[[ ## advice ## ]]")[-1].split("[[ ## completed")[0].strip()
        found = re.search(r"\{.*\}", advice, re.DOTALL)     # the advice is a dict, one entry per predictor
        try:
            advice = " ".join(json.loads(found.group(0)).values())
        except (ValueError, AttributeError):
            pass
        hints.append(advice)
    return hints[since:]


# --- shared reporting --------------------------------------------------------

CHECKS = ["category", "urgency", "grounded", "actionable"]


def evaluate_inbox(program, testset, num_threads: int = 3):
    """Score on the test split and break the score into the four checks.

    Returns (score, per_check_totals, rows). Re-scoring each prediction with
    score_inbox hits the judge cache, so the breakdown costs nothing extra.
    """
    evaluator = dspy.Evaluate(devset=testset, metric=inbox_metric, num_threads=num_threads,
                              display_progress=False)
    result = evaluator(program)
    totals = dict.fromkeys(CHECKS, 0)
    rows = []
    for example, prediction, score in result.results:
        checks = score_inbox(example, prediction)
        for name, ok in checks.items():
            totals[name] += int(ok)
        rows.append((example, prediction, score, checks))
    return result.score, totals, rows


def run_test(program, testset, recorder: Recorder | None = None) -> dict:
    """Score one program on the test split and count what the run used.

    `attempts` is drafts per email. A program with no retry loop makes one.
    `hints` is calls to Refine's hint writer per email.
    `redirects` is how many final replies tell the customer to email support.
    `judged` is judge calls per email made by the reward function.
    """
    hints_before = len(hints_written())
    score, totals, rows = evaluate_inbox(program, testset)
    n = len(testset)
    tried = [a for attempts in recorder.attempts.values() for a in attempts] if recorder else []
    attempts = len(tried) / n if recorder else 1.0
    judged = 2 * sum(not redirects_customer(reply) for _, reply in tried) / n
    return {
        "score": score, "totals": totals, "rows": rows,
        "redirects": sum(redirects_customer(pred.reply) for _, pred, _, _ in rows),
        "attempts": attempts, "judged": judged,
        "hints": (len(hints_written()) - hints_before) / n,
    }


def wrapped(label: str, text: str) -> str:
    """One labelled paragraph, wrapped at 84 columns so it fits a slide."""
    return textwrap.fill(text, width=84, initial_indent=label, subsequent_indent=" " * len(label))


def print_attempt_grid(rows, recorder: Recorder):
    """One row per email: the reward of each attempt. A blank cell means the loop had stopped."""
    print("attempt 1  attempt 2  attempt 3 | email")
    for example, _, _, _ in rows:
        rewards = [f"{reward:.2f}" for reward, _ in recorder.attempts[example.email]]
        cells = "  ".join(f"{cell:>9s}" for cell in rewards + [""] * (3 - len(rewards)))
        print(f"{cells} | {example.email[:48]}")


def print_result(label: str, r: dict, n: int = 6, header: bool = False):
    """One table row per program. Pass header=True on the first row."""
    if header:
        print(f"{'program':26s} {'score':>5s} {'category':>9s} {'urgency':>8s} {'grounded':>9s} "
              f"{'actionable':>11s} {'redirects':>10s}")
    t = r["totals"]
    cells = [f"{t[name]}/{n}" for name in ("category", "urgency", "grounded", "actionable")]
    print(f"{label:26s} {r['score']:>5.1f} {cells[0]:>9s} {cells[1]:>8s} {cells[2]:>9s} "
          f"{cells[3]:>11s} {str(r['redirects']) + '/' + str(n):>10s}")
