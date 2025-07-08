import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
try:
    lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))
    dspy.configure(lm=lm)
except Exception as e:
    print(f"Failed to initialize LM: {e}")
    exit()

lm("Hello, world!")
lm("What is DSPy?")

# Inspect recent interactions
print("--- History Inspection ---")
lm.inspect_history(n=2)

# Access programmatically
print("\n--- Programmatic History Access ---")
print(f"Total interactions: {len(lm.history)}")
for i, entry in enumerate(lm.history):
    cost_info = entry.get('response', {}).get('usage', {}).get('total_cost', 'N/A')
    print(f"Interaction {i+1}: Prompt='{entry['prompt']}', Cost=${cost_info}")
