import dspy
import os

def build_chatbot():
    # Note: set OPENROUTER_API_KEY environment variable
    try:
        lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"Failed to initialize the language model: {e}")
        print("Please ensure your OPENROUTER_API_KEY is set as an environment variable.")
        return

    print("DSPy Chatbot (type 'quit' to exit)")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            
            response = lm(f"You are a helpful assistant. User says: {user_input}")
            print(f"Bot: {response[0]}")
        except Exception as e:
            print(f"An error occurred during the conversation: {e}")
            break

if __name__ == "__main__":
    build_chatbot()
