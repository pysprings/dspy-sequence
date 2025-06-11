from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy
from utils import load_gsm8k, evaluate
from openai import OpenAI
from typing import TypedDict
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathState(TypedDict):
    question: str
    answer: str | None
    attempts: int


def solve_math(state: MathState):
    """Enhanced math solver with better prompting"""
    client = OpenAI()

    try:
        # Enhanced prompt for better math solving
        enhanced_prompt = f"""
        Solve this math problem step by step and provide only the final numerical answer.
        
        Problem: {state["question"]}
        
        Show your work, then end with: ANSWER: [number]
        """

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise math solver. Show work and give clear numerical answers.",
                },
                {"role": "user", "content": enhanced_prompt},
            ],
            max_tokens=150,
            temperature=0,  # Deterministic for math
        )

        full_response = response.choices[0].message.content.strip()

        # Extract the final answer
        answer = extract_final_answer(full_response)

        logger.info(f"Solved after {state.get('attempts', 0) + 1} attempts: {answer}")

        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    except Exception as e:
        logger.error(f"Error in solve_math: {str(e)}")
        return {"answer": f"ERROR: {str(e)}", "attempts": state.get("attempts", 0) + 1}


def extract_final_answer(response: str) -> str:
    """Extract numerical answer from response"""
    import re

    # Look for "ANSWER:" pattern first
    answer_pattern = r"ANSWER:\s*([+-]?\d*\.?\d+)"
    match = re.search(answer_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1)

    # Fallback patterns
    patterns = [
        r"Final Answer:\s*([+-]?\d*\.?\d+)",
        r"= ([+-]?\d*\.?\d+)(?:\s|$)",
        r"Answer:\s*([+-]?\d*\.?\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)

    # Last resort: find last number
    numbers = re.findall(r"[+-]?\d*\.?\d+", response)
    return numbers[-1] if numbers else "ERROR"


def langgraph_predict(question):
    """Enhanced LangGraph prediction with retry"""

    # Create retry policy for API failures
    retry_policy = RetryPolicy(
        max_attempts=3, backoff_factor=2.0, max_interval=30.0, jitter=True
    )

    workflow = StateGraph(MathState)

    # Add node with retry policy
    workflow.add_node("solver", solve_math, retry=retry_policy)

    workflow.set_entry_point("solver")
    workflow.add_edge("solver", END)

    # Compile
    app = workflow.compile()

    state = MathState(question=question, answer=None, attempts=0)

    try:
        result = app.invoke(state)

        class Prediction:
            def __init__(self, answer):
                self.answer = answer

        return Prediction(result["answer"])

    except Exception as e:
        logger.error(f"Final error in langgraph_predict: {str(e)}")

        class Prediction:
            def __init__(self, answer):
                self.answer = answer

        return Prediction("ERROR")


def main():
    print("=== Testing Enhanced LangGraph Math Solver ===")

    # Test with a few examples
    test_questions = [
        "What is 15% of 240?",
        "What is 25 × 16?",
        "If a rectangle has length 12 and width 8, what is its area?",
    ]

    print("Individual tests:")
    for question in test_questions:
        result = langgraph_predict(question)
        print(f"Q: {question}")
        print(f"A: {result.answer}")
        print()

    # Run evaluation
    print("\n=== Evaluation ===")
    testset = load_gsm8k()
    accuracy = evaluate(langgraph_predict, testset)
    print(f"Enhanced LangGraph Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
