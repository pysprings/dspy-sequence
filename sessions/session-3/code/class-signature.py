"""Class-based signatures (§4): the full power form.

Shows how a docstring + InputField/OutputField become the task contract.
Docstring → task instructions, desc → LM guidance per field.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


predictor = dspy.Predict(EnglishToUnix)

for request in [
    "list all files in the current directory",
    "find every Python file under src/",
    "count lines in README.md",
]:
    result = predictor(request=request)
    print(f"{request!r:55s} -> {result.command}")

# Peek at how the signature becomes instructions.
print("\n--- compiled task instructions ---")
print(EnglishToUnix.instructions)

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
