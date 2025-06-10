from datasets import load_dataset
import dspy

def load_gsm8k(split='test', n=10, offset=0):
    ds = load_dataset("openai/gsm8k", "main")
    dataset = []
    data = ds[split].select(range(offset, offset+n))
    for example_dict in data:
        true_answer = example_dict["answer"].split("####")[-1].strip()
        example = dspy.Example(question=example_dict["question"], answer=true_answer).with_inputs("question")
        dataset.append(example)
    return dataset

def evaluate(predict_fn, testset):
    correct = 0
    for example in testset:
        prediction = predict_fn(example.question)
        if str(prediction.answer) == str(example.answer):
            correct += 1
    return correct / len(testset)
