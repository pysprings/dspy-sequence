"""Your first metric (§3): a plain function, and the machine that runs it.

A metric is not a class or a decorator. It is a Python function with the
shape `metric(example, pred, trace=None)` returning a score. dspy.Evaluate
runs it across a devset and aggregates: booleans become a percentage.
"""

import dspy

dspy.configure(lm=dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


# A small devset in the Session 2 shape: gold fields, inputs marked.
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


def exact_command(example, pred, trace=None):
    """Tier 1 of the sline metric: character-for-character (after strip)."""
    return example.command.strip() == pred.command.strip()


program = dspy.Predict(EnglishToUnix)

evaluator = dspy.Evaluate(
    devset=devset,
    metric=exact_command,
    num_threads=4,
    display_progress=True,
)

result = evaluator(program)

print("\n--- what Evaluate returns ---")
print(f"type          : {type(result).__name__}")
print(f"result.score  : {result.score}   (percentage: 100 * mean of the scores)")
print(f"len(results)  : {len(result.results)}  (example, prediction, score) triples")

print("\n--- the misses, read by hand ---")
for example, prediction, score in result.results:
    if not score:
        print(f"request  : {example.request}")
        print(f"gold     : {example.command}")
        print(f"predicted: {prediction.command}")
        print()
