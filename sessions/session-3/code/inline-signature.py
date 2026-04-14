"""Inline signatures (§3): the `lambda` of DSPy signatures.

Demonstrates string-based signatures with different shapes — single
input/output, multi-input, multi-output, and type-annotated variants.
Run with OPENAI_API_KEY exported.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))

# One input, one output.
qa = dspy.Predict("question -> answer")
print("--- question -> answer")
print(qa(question="What is the capital of France?").answer)

# Multiple inputs.
rag = dspy.Predict("context, question -> answer")
print("\n--- context, question -> answer")
print(
    rag(
        context="The Nile is the longest river in the world, at 6,650 km.",
        question="Which river is longest, and how long is it?",
    ).answer
)

# Multiple outputs (reasoning before answer).
reasoned = dspy.Predict("question -> reasoning, answer")
out = reasoned(
    question="If a shirt costs $20 after a 20% discount, what was the original price?"
)
print("\n--- question -> reasoning, answer")
print("Reasoning:", out.reasoning)
print("Answer:   ", out.answer)

# Typed fields: DSPy coerces the output to the declared type.
typed = dspy.Predict("question: str -> answer: str, confidence: float")
out = typed(question="Is the Great Wall of China visible from low Earth orbit?")
print("\n--- question: str -> answer: str, confidence: float")
print("Answer:    ", out.answer)
print(f"Confidence: {out.confidence} (type={type(out.confidence).__name__})")

# Uncomment to see every compiled prompt + raw LM response for the 4 calls above:
# dspy.inspect_history(n=4)
