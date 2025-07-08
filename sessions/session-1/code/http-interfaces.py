import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
try:
    lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))
    dspy.configure(lm=lm)
except Exception as e:
    print(f"Failed to initialize LM: {e}")
    exit()

# Basic completion (prompt is a string)
response = lm("Tell me a story about a brave python.")
print("--- Basic Completion ---")
print(response[0])

# Chat format (using messages list)
chat_response = lm(messages=[
    {"role": "system", "content": "You are a helpful assistant that speaks in rhymes."},
    {"role": "user", "content": "Hello! What is your name?"}
])
print("\n--- Chat Completion ---")
print(chat_response[0])

# Streaming (will be covered in advanced sessions)
# print("\n--- Streaming ---")
# try:
#     # The stream method might not be directly available on the base LM object in all versions
#     # and is often used within specific DSPy modules. This is a conceptual example.
#     if hasattr(lm, 'stream'):
#         for chunk in lm.stream("Write a long poem about the sea."):
#             print(chunk, end="")
#     else:
#         print("lm.stream() not available on this object. Streaming is typically handled by modules.")
# except Exception as e:
#     print(f"Streaming failed: {e}")
