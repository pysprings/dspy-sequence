"""Custom DSPy adapter that uses HTML tags for the entire interaction.

Subclasses ``StructuredOutputAdapter`` -- which itself subclasses the
bare ``dspy.adapters.base.Adapter``. The only differences from the
YAML adapter are the three subclass methods: HTML tag scaffolding,
tag-wrapped encoding, and BeautifulSoup parsing.
"""

from pprint import pprint

import dspy
from bs4 import BeautifulSoup
from structured_output_adapter import StructuredOutputAdapter


class HtmlSoupAdapter(StructuredOutputAdapter):
    name = "HTML"

    def shape(self, fields):
        return "\n".join(f"<{name}>...</{name}>" for name in fields)

    def encode(self, values):
        return "\n".join(f"<{k}>{v}</{k}>" for k, v in values.items())

    def decode(self, completion, output_names):
        soup = BeautifulSoup(completion, "html.parser")
        return {
            name: "\n".join(soup.find(name).stripped_strings) for name in output_names
        }


class AnalyzeReview(dspy.Signature):
    """Analyze a product review."""

    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, negative, or mixed")
    key_points: str = dspy.OutputField(desc="bullet list of the main points raised")
    recommendation: str = dspy.OutputField(desc="should we follow up? yes or no")


dspy.configure(
    lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=400),
    adapter=HtmlSoupAdapter(),
)

sample_review = (
    "The headphones sound fantastic and the battery easily lasts a full workday, "
    "but the ear cushions started peeling after only two months and the companion "
    "app crashes every time I open it on Android."
)

result = dspy.Predict(AnalyzeReview)(review=sample_review)

print(f"sentiment:       {result.sentiment}")
print(f"key_points:      {result.key_points}")
print(f"recommendation:  {result.recommendation}")

print("\n--- result as dict ---")
pprint(dict(result))

print("\n--- compiled prompt + raw response ---")
dspy.inspect_history(n=1)
