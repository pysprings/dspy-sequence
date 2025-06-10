import dspy
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=1000))
solve = dspy.Predict("question -> answer: int")

question = "If Sarah has 15 apples and gives away 3 apples to each of her 4 friends, how many apples does she have left?"
result = solve(question=question)

print(f"Question: {question}")
print(f"Answer: {result.answer}")
print(dspy.inspect_history())
