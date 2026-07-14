"""Self-consistency (§7): n samples, one vote.

One ChainOfThought, five sampled completions, and dspy.majority to pick
the answer they agree on. Reasoning paths differ; correct answers converge.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=400))

# Passing n=5 as config makes the single LM call return five completions.
# Predict notices n > 1 and, if temperature is unset or <= 0.15, silently
# bumps it to 0.7 (predict.py); identical samples would make voting useless,
# so sampling needs heat.
cot = dspy.ChainOfThought("question -> answer", n=5)

question = (
    "A bakery sells muffins in boxes of 6. "
    "If they baked 87 muffins, how many full boxes can they sell?"
)

result = cot(question=question)

# The Prediction still has a single .answer (the first completion), but it
# also carries .completions, where each field is a list, one entry per sample.
print("--- five sampled answers ---")
for i, answer in enumerate(result.completions.answer):
    print(f"sample {i + 1}  : {answer}")

# majority() is a function, not a module; the zoo has both. It normalizes
# each completion's target field, counts, and returns a Prediction built from
# the first completion that voted with the winning bloc.
best = dspy.majority(result)
print("\n--- majority vote ---")
print(best.answer)

assert "14" in best.answer, best.answer

# Five cheap samples plus a vote often beats one careful sample, the same
# idea Session 7's optimizers exploit when they search over many candidates.

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
