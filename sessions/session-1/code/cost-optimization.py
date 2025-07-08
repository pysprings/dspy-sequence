import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
dspy.configure(api_key=os.environ.get("OPENROUTER_API_KEY"))

# Use cost-optimized routing
lm = dspy.LM('openrouter/meta-llama/llama-3.2-3b-instruct:floor')  # :floor = cheapest
print(f"Cost-optimized LM: {lm.model}")

# Free models for experimentation
free_lm = dspy.LM('openrouter/meta-llama/llama-3.2-3b-instruct:free')
print(f"Free-tier LM: {free_lm.model}")


# Speed-optimized routing
fast_lm = dspy.LM('openrouter/openai/gpt-4o-mini:nitro')  # :nitro = fastest
print(f"Speed-optimized LM: {fast_lm.model}")
