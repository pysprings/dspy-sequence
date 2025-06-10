import dspy
from datasets import load_dataset

# Load dataset
ds = load_dataset("openai/gsm8k", "main")
testset = [
    dspy.Example(question=x["question"], answer=x["answer"]).with_inputs("question")
    for x in ds["test"]
][:10]  # Use first 10 test examples

# Define metric
def exact_match(example, prediction, trace=None):
    # Extract integer from answer string
    try:
        pred_answer = prediction.answer.split("####")[-1].strip()
        true_answer = example.answer.split("####")[-1].strip()
        return pred_answer == true_answer
    except:
        return False

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=1000))

# Create solver
solve = dspy.Predict("question -> answer: int")

# Evaluate on testset
correct = 0
total = len(testset)

for example in testset:
    question = example.question
    prediction = solve(question=question)
    if exact_match(example, prediction):
        correct += 1

# Print evaluation results
print(f"Evaluation results:")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {correct/total:.2f}")

# Print history for last example
print("\nLast example history:")
print(dspy.inspect_history())
