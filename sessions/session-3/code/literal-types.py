"""Literal types (§5): constrained classification.

`Literal[...]` pins the output to an enumerated set. DSPy tells the LM the
allowed values and validates the response against them.
"""

from typing import Literal

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a text."""

    text: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()


classifier = dspy.Predict(SentimentClassifier)

samples = [
    "This product changed my life. Can't imagine going back.",
    "Meh. It does what it says on the tin, nothing more.",
    "Absolute trash. Do not buy.",
    "Shipping was fast, packaging was standard.",
]

for text in samples:
    result = classifier(text=text)
    assert result.sentiment in {"positive", "negative", "neutral"}, result.sentiment
    print(f"[{result.sentiment:8s}] {text}")

# Uncomment to see how `Literal[...]` is rendered as a prompt constraint:
# dspy.inspect_history(n=1)
