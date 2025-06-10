import dspy
from datasets import load_dataset
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Load dataset
ds = load_dataset("openai/gsm8k", "main")
trainset = [
    dspy.Example(question=x["question"], answer=x["answer"]).with_inputs("question")
    for x in ds["train"]
][:50]
devset = [
    dspy.Example(question=x["question"], answer=x["answer"]).with_inputs("question")
    for x in ds["test"]
][:10]


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


# Define program
class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("question -> answer: int")

    def forward(self, question):
        return self.solve(question=question)


# Optimize program
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
    num_threads=4,
)

math_solver = MathSolver()
optimized_solver = teleprompter.compile(math_solver, trainset=trainset, valset=devset)

# Save optimized program
optimized_solver.save("optimized_math_solver.json")

# Test optimized program
test_question = "If Sarah has 15 apples and gives away 3 apples to each of her 4 friends, how many apples does she have left?"
result = optimized_solver(question=test_question)

print(f"Question: {test_question}")
print(f"Answer: {result.answer}")
print(dspy.inspect_history())
