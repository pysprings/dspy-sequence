"""List + Optional output types (§5).

`list[str]` parses into a Python list. `Optional[str]` lets the LM skip the
field when it doesn't apply.
"""

from typing import Optional

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=400))


class EntityExtractor(dspy.Signature):
    """Extract named entities from text."""

    text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="named entities found in the text")
    summary: Optional[str] = dspy.OutputField(
        desc="brief summary if the text is longer than one sentence, otherwise null"
    )


extractor = dspy.Predict(EntityExtractor)

short = "Marie Curie discovered polonium."
long = (
    "In 1969, NASA's Apollo 11 mission landed Neil Armstrong and Buzz Aldrin on "
    "the Moon while Michael Collins orbited above. The Lunar Module, named Eagle, "
    "touched down in the Sea of Tranquility. The three astronauts returned safely "
    "aboard the Columbia command module."
)

for text in (short, long):
    result = extractor(text=text)
    print(f"text: {text}")
    print(f"  entities ({type(result.entities).__name__}): {result.entities}")
    print(f"  summary: {result.summary!r}\n")

# Uncomment to see both prompts side-by-side (note how list[str] and Optional render):
# dspy.inspect_history(n=2)
