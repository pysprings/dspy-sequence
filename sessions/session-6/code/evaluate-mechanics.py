"""dspy.Evaluate mechanics (§4): tables, saved runs, and failure handling.

Evaluate is a small machine with useful knobs:
  display_table=N   render N result rows (needs pandas)
  save_as_csv/json  persist every row for later diffing
  failure_score     what a crashed example scores (default 0.0)
  max_errors        how many crashes to tolerate before aborting
"""

import dspy

dspy.configure(lm=dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


devset = [
    dspy.Example(request=r, command=c).with_inputs("request")
    for r, c in [
        ("print the current working directory", "pwd"),
        ("list files including hidden ones", "ls -la"),
        ("count lines in notes.txt", "wc -l notes.txt"),
        ("show the last 20 lines of app.log", "tail -n 20 app.log"),
        ("make a directory called backups", "mkdir backups"),
    ]
]


def exact_command(example, pred, trace=None):
    return example.command.strip() == pred.command.strip()


program = dspy.Predict(EnglishToUnix)

evaluator = dspy.Evaluate(
    devset=devset,
    metric=exact_command,
    num_threads=4,
    display_progress=True,
    display_table=5,                  # render the first 5 rows
    save_as_csv="eval-run.csv",       # every row, for diffing across runs
    failure_score=0.0,                # a crashed example scores 0, not NaN
    max_errors=3,                     # tolerate 3 crashes, then abort
)

result = evaluator(program)

print(f"\nresult.score: {result.score}")

# The saved CSV has one row per example: inputs, gold, prediction, score.
with open("eval-run.csv") as f:
    header, first = f.readline(), f.readline()
print(f"csv header  : {header.strip()}")
print(f"csv row 1   : {first.strip()[:80]}...")

# Sort the triples to read the worst cases first: the daily-driver move.
worst = sorted(result.results, key=lambda triple: triple[2])
print("\n--- worst first ---")
for example, prediction, score in worst[:3]:
    print(f"{score}  gold: {example.command!r}  got: {prediction.command!r}")
