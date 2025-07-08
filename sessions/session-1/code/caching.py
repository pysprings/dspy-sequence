import dspy
import os
import time

# Caching enabled by default - saves money!
# Note: To see caching in action, you might need to clear litellm cache
# between runs. It's often in ~/.litellm_cache
# Note: set OPENROUTER_API_KEY environment variable
try:
    lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"), cache=True)
    dspy.configure(lm=lm)
except Exception as e:
    print(f"Failed to initialize LM: {e}")
    exit()


# First call - hits the API
print("First call:")
start_time = time.time()
response1 = lm("What is 2+2?")
end_time = time.time()
print(response1)
print(f"Duration: {end_time - start_time:.2f}s")


# Second identical call - returns cached result (free and fast!)
print("\nSecond call (should be cached):")
start_time = time.time()
response2 = lm("What is 2+2?")  # No API call made
end_time = time.time()
print(response2)
print(f"Duration: {end_time - start_time:.4f}s")

lm.inspect_history(n=2)
