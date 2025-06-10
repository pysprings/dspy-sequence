import dspy

# Configure the language model
dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=1000))


# Create a simple math solver using current API
class MathSolver(dspy.Module):
    def __init__(self):
        # Current API uses string signatures
        self.solve = dspy.ChainOfThought("question -> answer: int")

    def forward(self, question):
        return self.solve(question=question)


# Test the basic solver
solver = MathSolver()
result = solver(
    "If Sarah has 15 apples and gives away 3 apples to each of her 4 friends, how many apples does she have left?"
)

print(
    f"Question: If Sarah has 15 apples and gives away 3 apples to each of her 4 friends, how many apples does she have left?"
)
print(f"Answer: {result.answer}")
