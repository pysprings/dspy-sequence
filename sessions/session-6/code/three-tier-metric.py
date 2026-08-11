"""The sline three-tier metric (§5): exact, normalized, then an LLM judge.

Each tier catches what the cheaper one misses:
  tier 1  exact match       free        identical strings
  tier 2  normalized match  ~free       quote/whitespace differences (shlex)
  tier 3  LLM judge         one call    functional equivalence

The graded score (1.0 / 0.8 / judge-pass 1.0 / 0.0) rewards partial credit
at evaluation time. Adapted from case-studies/sline/code/metrics-example.py,
with the judge rewritten as a DSPy program instead of a raw prompt string.
"""

import shlex

import dspy

dspy.configure(lm=dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=200))

# A stronger model for judging than for the task: cheap worker, careful grader.
judge_lm = dspy.LM("openrouter/openai/gpt-4.1-mini", max_tokens=600)


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


def command_match(example, pred, trace=None):
    """Three tiers, cheapest first. Graded when scoring, strict when tracing."""
    expected, predicted = example.command, pred.command

    if exact_match(expected, predicted):
        score = 1.0
    elif normalized_match(expected, predicted):
        score = 0.8
    else:
        with dspy.context(lm=judge_lm):
            verdict = judge(request=example.request,
                            expected=expected, predicted=predicted)
        score = 1.0 if verdict.equivalent else 0.0

    if trace is not None:          # bootstrapping (Session 7): pass/fail only
        return score >= 0.8
    return score                   # evaluation: partial credit is signal


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


devset = [
    dspy.Example(request=r, command=c).with_inputs("request")
    for r, c in [
        ("print the current working directory", "pwd"),
        ("list files including hidden ones", "ls -la"),
        ("show disk usage of each top-level directory, human readable",
         "du -h --max-depth=1 /"),
        ("count lines in notes.txt", "wc -l notes.txt"),
        ("find all python files under the current directory",
         "find . -name '*.py'"),
        ("show the last 20 lines of app.log", "tail -n 20 app.log"),
        ("search for TODO in all python files", "grep TODO *.py"),
        ("make a directory called backups", "mkdir backups"),
    ]
]

program = dspy.Predict(EnglishToUnix)

for name, metric in [("exact only", lambda e, p, trace=None: float(
        exact_match(e.command, p.command))),
                     ("three-tier", command_match)]:
    evaluator = dspy.Evaluate(devset=devset, metric=metric, num_threads=4)
    result = evaluator(program)
    print(f"{name:11s}: {result.score}")

print("\n--- where each miss was rescued (or not) ---")
evaluator = dspy.Evaluate(devset=devset, metric=command_match, num_threads=4)
for example, prediction, score in evaluator(program).results:
    if score < 1.0 or not exact_match(example.command, prediction.command):
        print(f"score {score:>4}  gold: {example.command!r:34s} "
              f"predicted: {prediction.command!r}")
