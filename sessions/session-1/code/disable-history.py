import dspy
import os

# Disable global history tracking for production to prevent memory leaks
dspy.settings.disable_history = True

# Note: set OPENROUTER_API_KEY environment variable
try:
    lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))
    dspy.configure(lm=lm)
except Exception as e:
    print(f"Failed to initialize LM: {e}")
    exit()


lm("This call will not be tracked in lm.history")

print(f"History length: {len(lm.history)}") # should be 0
