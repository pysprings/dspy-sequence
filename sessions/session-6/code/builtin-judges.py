"""Built-in metrics (§8): what ships with dspy.evaluate.

Rule-based:  answer_exact_match, answer_passage_match, normalize_text
LLM judges:  SemanticF1, CompleteAndGrounded -- these are dspy.Module
             subclasses, so the judge is a program you can inspect.

SemanticF1 expects fields named `question`/`response` on the example and
`response` on the prediction: it grades long-form answers by key facts
covered (recall) and absence of padding (precision).
"""

import dspy
from dspy.evaluate import SemanticF1, answer_exact_match, normalize_text

dspy.configure(lm=dspy.LM("openrouter/openai/gpt-4.1-mini", max_tokens=1000))

# --- the rule-based family -------------------------------------------------

print("--- normalize_text: the shared normalizer ---")
for s in ["  The Answer!  ", "answer", "An ANSWER."]:
    print(f"{s!r:18s} -> {normalize_text(s)!r}")

example = dspy.Example(question="capital of France?", answer="Paris")
for guess in ["Paris", "  the PARIS. ", "Lyon"]:
    pred = dspy.Prediction(answer=guess)
    print(f"answer_exact_match({guess!r:16s}) = "
          f"{answer_exact_match(example, pred)}")

# --- the judge family ------------------------------------------------------

metric = SemanticF1(decompositional=True)

example = dspy.Example(
    question="Why do DSPy modules declare predictors in __init__?",
    response=(
        "Because assignment is registration: DSPy discovers tunable "
        "predictors by walking the module's attributes, so optimizers, "
        "save, and set_lm can find them. Predictors created inside "
        "forward are invisible to all three."
    ),
).with_inputs("question")

pred = dspy.Prediction(
    response=(
        "Declaring predictors as attributes in __init__ lets DSPy find "
        "them by walking attributes, which optimizers need."
    ),
)

# The judge returns a Prediction holding its score, like any module would.
score = metric(example, pred).score
print(f"\nSemanticF1 (eval mode)      : {score:.3f}")

# The same object, asked the bootstrapping question instead:
decision = metric(example, pred, trace=[]).score
print(f"SemanticF1 (trace mode)     : {decision}  (score >= 0.66?)")

# The judge is a program: read its actual conversation like any predictor's.
print("\n--- the judge's own prompt (tail) ---")
lines = dspy.settings.lm.history[-1]["messages"][-1]["content"]
print(lines[:400], "...")
