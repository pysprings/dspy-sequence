"""ChainOfThought under the hood (§4): same contract, different strategy.

ChainOfThought is composition, not magic: it wraps a plain Predict whose
signature has a "reasoning" output field prepended. The caller's contract
is unchanged -- same inputs in, answer out, plus reasoning on the side.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=400))

# Identical signature both ways -- the module choice is the strategy choice.
sig = "question -> answer"
plain = dspy.Predict(sig)
cot = dspy.ChainOfThought(sig)

print("--- what CoT actually holds ---")
# Inside the module is just another Predict, assigned to self.predict.
print(f"inner predictor : {cot.predict}")
# Its signature is ours, with a reasoning output field prepended.
print(f"inner signature : {cot.predict.signature}")
# named_predictors() walks attributes and finds that one Predict.
print(f"predictor names : {[name for name, _ in cot.named_predictors()]}")

print("\n--- same call, both modules ---")
question = "A shirt costs $25 after a 20% discount. What was the original price?"

plain_result = plain(question=question)
print(f"plain answer  : {plain_result.answer}")

cot_result = cot(question=question)
print(f"cot reasoning : {cot_result.reasoning}")
print(f"cot answer    : {cot_result.answer}")

# The extra field is real output, not decoration -- and only CoT has it.
assert hasattr(cot_result, "reasoning"), cot_result
assert not hasattr(plain_result, "reasoning"), plain_result

# How did "reasoning" get onto the wire? It didn't -- not by CoT, anyway.
# CoT only edits the signature; the adapter (Session 4) renders the extra
# field into the prompt exactly like any other output field. CoT never
# touches wire format, which is why it works with every adapter.

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
