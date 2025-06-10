from datasets import load_dataset
import dspy

def load_gsm8k_testset(n=10):
    ds = load_dataset("openai/gsm8k", "main")
    testset = []
    test_data = ds["test"].select(range(n))
    for example_dict in test_data:
        true_answer = example_dict["answer"].split("####")[-1].strip()
        example = dspy.Example(question=example_dict["question"], answer=true_answer).with_inputs("question")
        testset.append(example)
    return testset

def evaluate(predict_fn, testset):
    correct = 0
    for example in testset:
        prediction = predict_fn(example.question)
        if str(prediction.answer) == str(example.answer):
            correct += 1
    return correct / len(testset)
