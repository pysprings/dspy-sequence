"""Basic type annotations (§5): types aren't just docs — DSPy parses them.

Shows float and bool outputs getting auto-coerced from the LM's text response.
"""

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class MovieReview(dspy.Signature):
    """Analyze a movie review."""

    review: str = dspy.InputField()
    rating: float = dspy.OutputField(desc="rating from 0.0 to 5.0")
    recommend: bool = dspy.OutputField(desc="whether to recommend this movie")


analyzer = dspy.Predict(MovieReview)

reviews = [
    "A masterpiece. Every shot is a painting, every line a haiku. I'll be thinking about this film for weeks.",
    "Two hours of my life I'll never get back. The plot made no sense and the acting was wooden.",
    "Decent popcorn flick. Not memorable, but you won't hate yourself for watching it.",
]

for review in reviews:
    result = analyzer(review=review)
    print(f"review: {review[:60]}...")
    print(f"  rating   : {result.rating}  (type={type(result.rating).__name__})")
    print(f"  recommend: {result.recommend} (type={type(result.recommend).__name__})\n")

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
