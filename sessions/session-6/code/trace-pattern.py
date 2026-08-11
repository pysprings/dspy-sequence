"""The trace argument (§7): one function, two behaviors.

At evaluation time DSPy calls your metric with trace=None and wants a graded
score. During bootstrapping (Session 7), DSPy runs your program with tracing
on and calls the metric with the trace to ask a different question: "is this
run clean enough to become a demonstration?" That answer should be strict.

No optimizer here; we call the metric both ways by hand to see the switch.
"""

import dspy

dspy.configure(lm=dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


def command_match(example, pred, trace=None):
    """Graded for evaluation; strict pass/fail for bootstrapping."""
    exact = example.command.strip() == pred.command.strip()
    same_program = example.command.split()[0] == pred.command.split()[0]

    if trace is not None:                 # bootstrapping: only exact is clean
        return exact
    return (exact + same_program) / 2.0   # evaluation: partial credit


example = dspy.Example(
    request="show the last 20 lines of app.log",
    command="tail -n 20 app.log",
).with_inputs("request")

close = dspy.Prediction(command="tail -20 app.log")     # right tool, off flag
perfect = dspy.Prediction(command="tail -n 20 app.log")

print("--- evaluation mode: trace is None, graded ---")
print(f"close   : {command_match(example, close)}")
print(f"perfect : {command_match(example, perfect)}")

print("\n--- bootstrap mode: trace passed, strict ---")
print(f"close   : {command_match(example, close, trace=[])}")
print(f"perfect : {command_match(example, perfect, trace=[])}")

# What is actually in a trace? Run the program with tracing on and look.
program = dspy.Predict(EnglishToUnix)
with dspy.context(trace=[]):
    pred = program(request=example.request)
    trace = dspy.settings.trace

print("\n--- inside a real trace ---")
for predictor, inputs, outputs in trace:
    print(f"predictor : {predictor.signature.signature}")
    print(f"inputs    : {inputs}")
    print(f"outputs   : {outputs}")

# During compilation, Session 7's bootstrapper does morally this:
#   passed = command_match(example, pred, trace)
#   if passed: keep (inputs, outputs) as a demo for the predictor
print("\nwould this run become a demo?",
      bool(command_match(example, pred, trace)))
