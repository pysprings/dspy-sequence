import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
try:
    lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))
except Exception as e:
    print(f"Failed to initialize LM: {e}")
    exit()

# Create a copy of the model with different parameters for testing
test_lm = lm.copy(temperature=0.8, max_tokens=100)

print("--- Original LM State ---")
print(lm.dump_state())

print("\n--- Copied LM State (for testing) ---")
# Inspect the new model's state
state = test_lm.dump_state()
print(state)
