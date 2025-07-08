import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
dspy.configure(api_key=os.environ.get("OPENROUTER_API_KEY"))

# If OpenAI goes down, OpenRouter automatically switches to another provider
# for the same model if available.
# This behavior is inherent to OpenRouter and doesn't require special DSPy code.
lm = dspy.LM('openrouter/mistralai/mistral-small-24b-isnstruct-2501')  # Transparent failover built-in
print(f"Initialized LM with potential failover: {lm.model}")

# You can test this by making a call.
# response = lm("What is automatic failover?")
# print(response)
