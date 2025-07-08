# With OpenRouter - one API for everything
import dspy
import os

# Set API key for OpenRouter
# Note: set OPENROUTER_API_KEY environment variable
dspy.configure(api_key=os.environ.get("OPENROUTER_API_KEY"))

# All models through OpenRouter's unified interface
try:
    openai_lm = dspy.LM('openrouter/openai/gpt-4o-mini')
    anthropic_lm = dspy.LM('openrouter/anthropic/claude-3-haiku-20240307')
    google_lm = dspy.LM('openrouter/google/gemini-1.5-flash')
    meta_lm = dspy.LM('openrouter/meta-llama/llama-3.2-3b-instruct')

    # Local models still use direct provider format
    # This will fail if Ollama is not running.
    local_lm = dspy.LM('ollama/llama3.2:1b')

    print("Successfully initialized LMs from OpenRouter and local providers.")
    print(f"OpenAI model: {openai_lm.model}")
    print(f"Anthropic model: {anthropic_lm.model}")
    print(f"Google model: {google_lm.model}")
    print(f"Meta model: {meta_lm.model}")
    print(f"Local model: {local_lm.model}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your OPENROUTER_API_KEY is set and Ollama is running.")
