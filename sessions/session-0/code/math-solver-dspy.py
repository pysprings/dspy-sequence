import dspy
from datasets import load_dataset

# Load dataset
ds = load_dataset("openai/gsm8k", "main")
testset = [
    dspy.Example(question=x["question"], answer=x["answer"]).with_inputs("question")
    for x in ds["test"]
][:10]  # Use first 10 test examples

# Define metric with debugging
def exact_match(example, prediction, trace=None):
    # Extract integer from answer string
    try:
        pred_answer = prediction.answer.split("####")[-1].strip()
        true_answer = example.answer.split("####")[-1].strip()
        match = pred_answer == true_answer
        if not match:
            print(f"\nMismatch for question: {example.question}")
            print(f"True answer: {example.answer} -> extracted: '{true_answer}'")
            print(f"Predicted answer: {prediction.answer} -> extracted: '{pred_answer}'")
        return match
    except Exception as e:
        print(f"\nError processing example: {example.question}")
        print(f"True answer: {example.answer}")
        print(f"Predicted answer: {prediction.answer}")
        print(f"Exception: {str(e)}")
        return False

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=1000))

# Create solver
solve = dspy.Predict("question -> answer: int")

# Evaluate on testset
correct = 0
total = len(testset)

for i, example in enumerate(testset):
    question = example.question
    prediction = solve(question=question)
    print(f"\nProcessing example {i+1}/{len(testset)}")
    print(f"Question: {question}")
    print(f"Prediction: {prediction.answer}")
    if exact_match(example, prediction):
        correct += 1
        print("✅ Correct")
    else:
        print("❌ Incorrect")

# Print evaluation results
print(f"Evaluation results:")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {correct/total:.2f}")

# Print history for last example
print("\nLast example history:")
print(dspy.inspect_history())
