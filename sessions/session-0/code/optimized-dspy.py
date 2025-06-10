from utils import load_gsm8k_testset, evaluate
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("question -> answer: int")
    
    def forward(self, question):
        return self.solve(question=question)

def main():
    testset = load_gsm8k_testset(10)
    dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=1000))
    
    # Optimize with BootstrapFewShotWithRandomSearch
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=lambda example, pred, trace=None: str(pred.answer) == str(example.answer),
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_candidate_programs=10,
        num_threads=4
    )
    
    solver = MathSolver()
    optimized_solver = teleprompter.compile(solver, trainset=testset[:5])
    
    accuracy = evaluate(optimized_solver, testset)
    print(f"Optimized DSPy Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
