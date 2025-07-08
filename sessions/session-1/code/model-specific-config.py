import dspy
import os

# OpenAI's reasoning models (like o1-mini) have special requirements
# for some parameters.
try:
    # Note: set OPENROUTER_API_KEY environment variable
    reasoning_lm = dspy.LM(
        'openrouter/openai/o1-mini',
        temperature=1.0,  # Must be 1.0 for this model
        max_tokens=20000,  # Must be >= 20,000 for this model
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
    print("Successfully created reasoning_lm instance.")
    print(reasoning_lm.dump_state())
except Exception as e:
    print(f"Failed to create reasoning_lm instance: {e}")
