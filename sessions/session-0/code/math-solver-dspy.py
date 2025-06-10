import dspy

# Configure the language model
dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=1000))


# Create a simple math solver using current API
solve = dspy.ChainOfThought("question -> answer: int")

# Test the basic solver
question = "If Sarah has 15 apples and gives away 3 apples to each of her 4 friends, how many apples does she have left?"
result = solve(question=question)

print(f"Question: {question}")
print(f"Answer: {result.answer}")
