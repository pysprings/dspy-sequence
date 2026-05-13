"""Signatures + Examples (§8): Sessions 2 and 3 together.

Field names in dspy.Example must match the signature. This alignment is
what lets optimizers (Session 7) improve a system automatically.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


trainset = [
    dspy.Example(
        request="list all files in the current directory", command="ls -la"
    ).with_inputs("request"),
    dspy.Example(request="show running processes", command="ps aux").with_inputs(
        "request"
    ),
    dspy.Example(
        request="find every Python file", command="find . -name '*.py'"
    ).with_inputs("request"),
]

# Examples match the signature's fields — validate that alignment.
sig_fields = set(EnglishToUnix.model_fields)
example_fields = set(trainset[0].keys())
assert sig_fields == example_fields, (sig_fields, example_fields)
print(f"signature fields : {sorted(sig_fields)}")
print(f"example fields   : {sorted(example_fields)}")
print(f"example inputs   : {trainset[0].inputs().keys()}")
print(f"example labels   : {trainset[0].labels().keys()}\n")

# Same signature, used with a module to actually generate.
generator = dspy.ChainOfThought(EnglishToUnix)
for ex in trainset:
    pred = generator(request=ex.request)
    match = "✓" if pred.command.strip() == ex.command else "✗"
    print(f"[{match}] {ex.request}")
    print(f"    gold      : {ex.command}")
    print(f"    predicted : {pred.command}")

# Uncomment to see the compiled prompt for the last call (no examples embedded yet —
# that's what optimizers in Session 7 do):
# dspy.inspect_history(n=1)
