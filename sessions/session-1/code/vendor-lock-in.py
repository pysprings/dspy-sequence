import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
api_key = os.environ.get("OPENROUTER_API_KEY")

try:
    # Development with a cheap, fast model
    dev_lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=api_key)

    # Production with a different, more powerful provider
    prod_lm = dspy.LM('openrouter/anthropic/claude-3-5-sonnet-20241022', api_key=api_key)
except Exception as e:
    print(f"Failed to initialize LMs: {e}")
    exit()

# The same code works with both!
# By default, let's use the dev model
dspy.configure(lm=dev_lm)
print(f"Running with default LM: {dspy.settings.lm.model}")
result_dev = dspy.Predict("question -> answer")(question="Why is the sky blue?")
print(f"Dev Answer: {result_dev.answer[:80]}...")


# Use a context block to temporarily switch to the production model
print(f"\nTemporarily switching to: {prod_lm.model}")
with dspy.context(lm=prod_lm):
    result_prod = dspy.Predict("question -> answer")(question="Why is the sky blue?")
    print(f"Prod Answer: {result_prod.answer[:80]}...")

print(f"\nBack to default LM: {dspy.settings.lm.model}")
