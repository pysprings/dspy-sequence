"""Predict vs ChainOfThought (§7): same signature, different strategy.

`ChainOfThought` prepends a `reasoning` OutputField to your signature —
your code is unchanged but the LM now thinks before answering.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=400))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


fast = dspy.Predict(EnglishToUnix)
thoughtful = dspy.ChainOfThought(EnglishToUnix)

request = "find files larger than 100MB modified in the last week"

fast_out = fast(request=request)
print("--- Predict ---")
print(f"command  : {fast_out.command}")
print(f"reasoning: {getattr(fast_out, 'reasoning', '<not present>')}")

thoughtful_out = thoughtful(request=request)
print("\n--- ChainOfThought ---")
print(f"command  : {thoughtful_out.command}")
print(f"reasoning: {thoughtful_out.reasoning}")

# Uncomment to see both prompts side-by-side — note the extra `reasoning` field
# ChainOfThought inserts in front of your declared outputs:
# dspy.inspect_history(n=2)
