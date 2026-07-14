"""Your first module (§5): wrap a predictor, get a program.

The standard minimal pattern (a Signature wrapped by a Module holding one
`dspy.Predict`) is exactly the shape of sline's ShellAssistantModule.
Assign predictors in __init__, do the work in forward, call the instance.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


class UnixAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        # Assignment IS registration: because self.predict is an attribute
        # holding a Predict, named_predictors() finds it; no decorator, no
        # explicit registry. Optimizers walk this list to tune each predictor.
        self.predict = dspy.Predict(EnglishToUnix)

    def forward(self, request: str) -> dspy.Prediction:
        return self.predict(request=request)


assistant = UnixAssistant()

# Call the instance, never forward() directly: Module.__call__ wraps forward
# with the callback, usage-tracking, and history plumbing DSPy relies on.
print("--- the module sees its predictors ---")
print(repr(assistant))
print([name for name, _ in assistant.named_predictors()])

print("\n--- it runs ---")
requests = [
    "show disk usage of each top-level directory, human readable",
    "grpe for TODO in python files",  # typo the LM should fix
    "count the lines in every .csv file here",
]
for request in requests:
    result = assistant(request=request)
    print(f"{request!r:48s} -> {result.command}")

# Why wrap one Predict at all? Because the Module is the unit of
# optimization (Session 7's compilers take a module, not a signature)
# and the unit of composition: the next section of this session chains
# several predictors inside one forward, and this skeleton is where they go.

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
