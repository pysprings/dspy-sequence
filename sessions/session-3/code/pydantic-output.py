"""Pydantic outputs (§5): full structured output.

DSPy tells the LM to produce JSON matching the schema, then parses and
validates the response into Pydantic instances.
"""

import dspy
from pydantic import BaseModel

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=500))


class Entity(BaseModel):
    name: str
    type: str
    confidence: float


class EntityExtractor(dspy.Signature):
    """Extract structured entities from text. Use types like 'person',
    'organization', 'location', 'spacecraft', or 'event'."""

    text: str = dspy.InputField()
    entities: list[Entity] = dspy.OutputField()


extractor = dspy.Predict(EntityExtractor)

text = (
    "In 1969, NASA's Apollo 11 mission landed Neil Armstrong and Buzz Aldrin on "
    "the Moon while Michael Collins orbited above in the Columbia command module."
)
result = extractor(text=text)

print(f"text: {text}\n")
print("entities:")
for entity in result.entities:
    assert isinstance(entity, Entity), type(entity)
    print(f"  {entity.name:20s} type={entity.type:15s} conf={entity.confidence:.2f}")

# Uncomment to see the JSON schema for Entity embedded in the compiled prompt:
# dspy.inspect_history(n=1)
