import dspy
import os

# For reproducible results in testing, set temperature to 0.0
# Note: set OPENROUTER_API_KEY environment variable
try:
    deterministic_lm = dspy.LM('openrouter/openai/gpt-4o-mini', temperature=0.0, api_key=os.environ.get("OPENROUTER_API_KEY"))
    dspy.configure(lm=deterministic_lm)
except Exception as e:
    print(f"Failed to initialize LM: {e}")
    exit()


# The same input will always produce the same output (when temperature=0.0)
prompt = "What are the first 5 prime numbers?"
print(f"Prompt: {prompt}")

print("\n--- First call ---")
response1 = deterministic_lm(prompt)
print(response1[0])

print("\n--- Second call (should be identical) ---")
response2 = deterministic_lm(prompt)
print(response2[0])

assert response1[0] == response2[0]
print("\nOutputs are identical.")
