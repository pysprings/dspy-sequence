"""Predict anatomy (§3): the atom of DSPy, a strategy plus state.

A `dspy.Predict` is just an object holding STATE (signature, demos, config)
and a minimal STRATEGY: one adapter-formatted LM call. Everything fancier
(ChainOfThought, ReAct) is built by composing this atom.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


predictor = dspy.Predict(EnglishToUnix)

# The state: everything the strategy needs to build one prompt. `demos` is
# the empty slot that optimizers fill with worked examples in Session 7;
# `config` holds per-call LM kwargs like temperature.
print("--- the state ---")
print(f"signature: {predictor.signature}")
print(f"demos    : {predictor.demos}")
print(f"config   : {predictor.config}")

# The call: keyword args matching the input fields, a Prediction back.
# Positional args raise: predictor("show disk usage") is a ValueError
# naming the keyword arguments to use, because inputs must map onto
# named signature fields.
print("\n--- plain call: stored state only ---")
plain = predictor(request="show disk usage of each top-level directory, human readable")
print(f"command  : {plain.command}")
print(f"type     : {type(plain).__name__}")
assert isinstance(plain, dspy.Prediction), plain

# Runtime override: `forward` pops the privileged kwargs (signature,
# demos, config, lm) before treating the rest as inputs. Injecting a demo
# here steers style without touching the predictor's stored state; this is
# exactly the knob optimizers turn permanently.
print("\n--- same request, demo injected for this call ---")
demo = dspy.Example(
    request="print the current working directory",
    command="pwd  # print working directory",
)
steered = predictor(
    request="show disk usage of each top-level directory, human readable",
    demos=[demo],
)
print(f"command  : {steered.command}")
print(f"stored demos after the call: {predictor.demos}  (the override did not persist)")

print("\n--- last prompt + completion ---")
dspy.inspect_history(n=1)
