"""Field constraints (§5): ge/le/max_length and friends.

Pydantic-style constraints on OutputField get compiled into human-readable
hints in the system prompt.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=400))


class BoundedScore(dspy.Signature):
    """Score the quality of a response on a 1-10 scale with brief justification."""

    response: str = dspy.InputField()
    score: int = dspy.OutputField(ge=1, le=10, desc="quality score")
    justification: str = dspy.OutputField(max_length=200)


scorer = dspy.Predict(BoundedScore)

samples = [
    "The capital of France is Paris, a city known for the Eiffel Tower and Louvre.",
    "yes",
    "Paris is the capital city of France. It sits on the river Seine in northern France.",
]

for response in samples:
    result = scorer(response=response)
    assert 1 <= result.score <= 10, result.score  # numeric bounds: enforced by parser
    within = "✓" if len(result.justification) <= 200 else "✗"
    print(f"[score={result.score:>2}] {response[:50]}...")
    print(
        f"         justification ({len(result.justification)} chars, "
        f"max_length=200 {within}): {result.justification}\n"
    )

# Constraints flow into the compiled system prompt — inspect how.
print("--- Last prompt sent to the LM ---")
dspy.inspect_history(n=1)
