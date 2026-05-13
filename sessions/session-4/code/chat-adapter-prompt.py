"""Show what the default ChatAdapter actually puts on the wire.

Runs the same EnglishToUnix signature from session 3 and prints the
compiled system + user messages, including the [[ ## field ## ]] markers
the adapter inserts. This is the prompt your signature became.
"""

from pprint import pprint

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


predictor = dspy.Predict(EnglishToUnix)
result = predictor(request="list all files in the current directory")
print(f"command: {result.command}")

print("\n--- result as dict ---")
pprint(dict(result))

print("\n--- what the ChatAdapter sent ---")
dspy.inspect_history(n=1)
