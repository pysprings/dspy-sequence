from langgraph.graph import StateGraph, END
from utils import load_gsm8k_testset, evaluate
import openai

# Simple state for LangGraph
class MathState:
    def __init__(self, question):
        self.question = question
        self.answer = None

def solve_math(state):
    response = openai.ChatCompletion.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Solve math problems. Answer with just the number."},
            {"role": "user", "content": state.question}
        ],
        max_tokens=50
    )
    state.answer = response.choices[0].message.content.strip()
    return state

def langgraph_predict(question):
    workflow = StateGraph(MathState)
    workflow.add_node("solver", solve_math)
    workflow.set_entry_point("solver")
    workflow.add_edge("solver", END)
    app = workflow.compile()
    
    state = MathState(question)
    result = app.invoke(state)
    
    # Simple class to mimic DSPy Prediction structure
    class Prediction:
        def __init__(self, answer):
            self.answer = answer
    return Prediction(result.answer)

def main():
    testset = load_gsm8k_testset(10)
    accuracy = evaluate(langgraph_predict, testset)
    print(f"LangGraph Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
