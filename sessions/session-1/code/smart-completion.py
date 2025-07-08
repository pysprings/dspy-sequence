import dspy
import os

def smart_completion(partial_command):
    # Note: set OPENROUTER_API_KEY environment variable
    try:
        lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))
    except Exception as e:
        print(f"Failed to initialize the language model: {e}")
        return "Error: Could not initialize LM."
    
    completion_prompt = f"""
    Complete this shell command: {partial_command}
    Only return the completed command, nothing else.
    """
    
    result = lm(completion_prompt)
    return result[0].strip()

if __name__ == "__main__":
    # Example usage
    completed_command = smart_completion("git commit -m ")
    print(f"Original: git commit -m \nCompleted: {completed_command}")
    # Example Output might be: git commit -m "feat: implement smart completion"
