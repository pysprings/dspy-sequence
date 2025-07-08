import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
dspy.configure(api_key=os.environ.get("OPENROUTER_API_KEY"))

# Start with a fast, cheap model for development
dev_lm = dspy.LM('openrouter/openai/gpt-4o-mini')
dspy.configure(lm=dev_lm)
print(f"Default LM configured to: {dspy.settings.lm.model}")


# Switch to a more powerful model for production
# Just change the string - no code changes needed!
prod_lm = dspy.LM('openrouter/anthropic/claude-3-5-sonnet-20241022')
dspy.configure(lm=prod_lm)
print(f"Default LM switched to: {dspy.settings.lm.model}")


# Or even switch to a local model for privacy
# This will fail if Ollama is not running.
try:
    local_lm = dspy.LM('ollama/llama3.2:3b')
    dspy.configure(lm=local_lm)
    print(f"Default LM switched to: {dspy.settings.lm.model}")
except Exception as e:
    print(f"\nCould not switch to local model. Is Ollama running? Error: {e}")
