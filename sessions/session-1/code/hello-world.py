import dspy
import os

# Create a language model instance
# Note: set OPENROUTER_API_KEY environment variable
lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))

# Configure DSPy to use this model globally
dspy.configure(lm=lm)

# Direct usage - simple prompt
response = lm("What is the capital of France?")
print(response)
