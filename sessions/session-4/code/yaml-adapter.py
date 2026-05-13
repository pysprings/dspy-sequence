"""Custom DSPy adapter that uses YAML for the entire interaction.

Subclasses ``StructuredOutputAdapter`` -- which itself subclasses the
bare ``dspy.adapters.base.Adapter`` -- so the only code in this file
is the three things that change for YAML: scaffold shape, value
encoding, and response decoding.
"""

from pprint import pprint

import yaml

import dspy
from structured_output_adapter import StructuredOutputAdapter


class YamlAdapter(StructuredOutputAdapter):
    name = "YAML"

    def shape(self, fields):
        return "\n".join(f"{name}: <{name}>" for name in fields)

    def encode(self, values):
        return yaml.safe_dump(
            values,
            sort_keys=False,
            allow_unicode=True,        # keep em-dashes etc. literal
            width=float("inf"),        # don't wrap long strings
            default_flow_style=False,
        ).strip()

    def decode(self, completion, output_names):
        return yaml.safe_load(completion)


class TriageEmail(dspy.Signature):
    """Triage an incoming support email."""

    email: str = dspy.InputField()
    category: str = dspy.OutputField(desc="one of: billing, technical, account, other")
    priority: str = dspy.OutputField(desc="one of: low, medium, high")
    summary: str = dspy.OutputField(desc="one-sentence summary of the issue")


dspy.configure(
    lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=300),
    adapter=YamlAdapter(),
)

sample_email = (
    "Subject: Charged twice this month\n\n"
    "Hi — I just noticed two identical $49 charges from you on my card "
    "for May. I only have one subscription. Can someone refund the duplicate?"
)

result = dspy.Predict(TriageEmail)(email=sample_email)

print(f"category: {result.category}")
print(f"priority: {result.priority}")
print(f"summary:  {result.summary}")

print("\n--- result as dict ---")
pprint(dict(result))

print("\n--- last prompt + completion ---")
dspy.inspect_history(n=1)
