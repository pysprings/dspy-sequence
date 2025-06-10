import dspy
from datasets import load_dataset

# Load dataset
ds = load_dataset("openai/gsm8k", "main")
# Preprocess answers to extract just the final number
testset = []
test_data = ds["test"].select(range(10))  # Get first 10 test examples
for example_dict in test_data:
    # Extract the final answer number from the answer string
    true_answer = example_dict["answer"].split("####")[-1].strip()
    example = dspy.Example(question=example_dict["question"], answer=true_answer).with_inputs("question")
    testset.append(example)


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
    
    # Convert both answers to strings for comparison
    true_answer_str = str(example.answer)
    pred_answer_str = str(prediction.answer)
    
    # Simple string comparison
    is_correct = true_answer_str == pred_answer_str
    if is_correct:
        correct += 1
        print("✅ Correct")
    else:
        print("❌ Incorrect")
        print(f"True answer: {true_answer_str}")
        print(f"Predicted answer: {pred_answer_str}")

# Print evaluation results
print(f"Evaluation results:")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {correct/total:.2f}")

# Print history for last example
print("\nLast example history:")
print(dspy.inspect_history())
