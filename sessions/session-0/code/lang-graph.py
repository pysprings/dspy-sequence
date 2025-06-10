from langgraph.graph import StateGraph, END
from utils import load_gsm8k_testset, evaluate
import openai
from typing import TypedDict

# Define state as TypedDict
class MathState(TypedDict):
    question: str
    answer: str | None

def solve_math(state: MathState):
    response = openai.ChatCompletion.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Solve math problems. Answer with just the number."},
            {"role": "user", "content": state["question"]}
        ],
        max_tokens=50
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer}

def langgraph_predict(question):
    workflow = StateGraph(MathState)
    workflow.add_node("solver", solve_math)
    workflow.set_entry_point("solver")
    workflow.add_edge("solver", END)
    app = workflow.compile()
    
    state = MathState(question=question, answer=None)
    result = app.invoke(state)
    
    # Simple class to mimic DSPy Prediction structure
    class Prediction:
        def __init__(self, answer):
            self.answer = answer
    return Prediction(result["answer"])

def main():
    testset = load_gsm8k_testset(10)
    accuracy = evaluate(langgraph_predict, testset)
    print(f"LangGraph Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
