import dspy
import os

# Note: set OPENROUTER_API_KEY environment variable
try:
    lm = dspy.LM('openrouter/openai/gpt-4o-mini', api_key=os.environ.get("OPENROUTER_API_KEY"))
    dspy.configure(lm=lm)
except Exception as e:
    print(f"Failed to initialize LM: {e}")
    exit()

# Define a simple classifier using the LM
classifier = dspy.Predict("text -> sentiment")
result = classifier(text="I love this product!")
print(f"Sentiment: {result.sentiment}")

# Create a custom module
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

qa_system = SimpleQA()
answer = qa_system("What is machine learning?")
print(f"Question: What is machine learning?")
print(f"Answer: {answer.answer}")
