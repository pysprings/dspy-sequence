"""Three-tier metrics for evaluating shell command predictions.

Demonstrates the tiered approach:
1. Exact match (free) - character-for-character
2. Normalized match (cheap) - tokenization handles quote/whitespace differences  
3. Semantic match (expensive) - LLM judge for functional equivalence

This pattern is useful whenever you have multiple valid answers
or need to handle formatting variations.
"""

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy


def exact_match(expected: str, predicted: str) -> bool:
    """Tier 1: Check if predicted command exactly matches expected.
    
    Cost: Zero - pure string comparison
    Catches: Identical outputs
    """
    return expected.strip() == predicted.strip()


def normalized_match(expected: str, predicted: str) -> bool:
    """Tier 2: Check if commands match after tokenization.
    
    Handles differences in quote styles, whitespace, etc.
    
    Cost: Negligible - shell tokenization only
    Catches: Quote style differences, extra whitespace
    
    Examples:
        normalized_match("grep 'foo' file.txt", 'grep "foo" file.txt')  # True
        normalized_match("ls  -la", "ls -la")  # True
    """
    try:
        expected_tokens = shlex.split(expected.strip())
        predicted_tokens = shlex.split(predicted.strip())
        return expected_tokens == predicted_tokens
    except ValueError:
        # shlex.split failed (unbalanced quotes, etc.)
        # Fall back to exact match
        return exact_match(expected, predicted)


def score(expected: str, predicted: str) -> float:
    """Compute a simple score between 0 and 1.
    
    - 1.0: exact match
    - 0.8: normalized match (same tokens, different formatting)
    - 0.0: no match
    
    Note: This simple scoring doesn't use LLM judge.
    Use semantic_match for full evaluation.
    """
    if exact_match(expected, predicted):
        return 1.0
    if normalized_match(expected, predicted):
        return 0.8
    return 0.0


def semantic_match(
    input_text: str,
    expected: str,
    predicted: str,
    judge_lm: "dspy.LM",
    alternates: list[str] | None = None,
) -> tuple[bool, str]:
    """Tier 3: Check if commands are semantically equivalent using LLM judge.
    
    Uses fast paths for exact/normalized matches first, then falls back to LLM.
    Results are cached automatically by DSPy.
    
    Args:
        input_text: The user's natural language request
        expected: The primary expected command
        predicted: The model's predicted command
        judge_lm: DSPy language model for semantic judging
        alternates: Optional list of alternate acceptable commands
    
    Returns:
        Tuple of (is_match, match_type) where match_type is one of:
        - "exact": exact string match
        - "normalized": same tokens, different formatting
        - "semantic": LLM judged as equivalent
        - "miss": not equivalent
    
    Cost: One LLM API call (only if fast paths fail)
    Catches: Functionally equivalent commands
    """
    # Build list of all acceptable answers
    acceptable = [expected] + (alternates or [])
    
    # Fast paths - no API call needed
    for ans in acceptable:
        if exact_match(ans, predicted):
            return True, "exact"
    
    for ans in acceptable:
        if normalized_match(ans, predicted):
            return True, "normalized"
    
    # LLM judge with chain-of-thought reasoning
    prompt = f"""You are evaluating whether two shell commands are semantically equivalent.

User's request: {input_text}
Expected command: {expected}
Predicted command: {predicted}

Think through this step by step:
1. What does the expected command do?
2. What does the predicted command do?  
3. Would they produce the same result for this user request?

After your analysis, answer with exactly YES or NO on a new line."""

    response = judge_lm(prompt)
    # Extract YES/NO from response (last word)
    answer = str(response[0]).strip().split()[-1].upper().rstrip(".")
    
    if answer == "YES":
        return True, "semantic"
    return False, "miss"


def make_metric(judge_lm: "dspy.LM"):
    """Create a metric function for DSPy optimization.
    
    Returns a function compatible with DSPy optimizers like MIPROv2.
    The metric returns 1.0 for matches, 0.0 for misses.
    """
    import dspy
    
    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        """Metric function for DSPy optimization using semantic matching."""
        expected = example.command
        predicted = prediction.command if hasattr(prediction, "command") else ""
        is_match, _ = semantic_match(example.request, expected, predicted, judge_lm)
        return 1.0 if is_match else 0.0
    
    return metric


# Example usage
if __name__ == "__main__":
    # Demonstrate the tiers without LLM
    print("=== Tier 1: Exact Match ===")
    print(f"'ls -la' vs 'ls -la': {exact_match('ls -la', 'ls -la')}")
    print(f"'ls -la' vs 'ls -l -a': {exact_match('ls -la', 'ls -l -a')}")
    
    print("\n=== Tier 2: Normalized Match ===")
    print(f"\"grep 'foo' f.txt\" vs 'grep \"foo\" f.txt': {normalized_match(\"grep 'foo' f.txt\", 'grep \"foo\" f.txt')}")
    print(f"'ls  -la' vs 'ls -la': {normalized_match('ls  -la', 'ls -la')}")
    print(f"'ls -la' vs 'ls -l -a': {normalized_match('ls -la', 'ls -l -a')}")
    
    print("\n=== Simple Score ===")
    print(f"score('ls -la', 'ls -la'): {score('ls -la', 'ls -la')}")
    print(f"score('ls -la', 'ls  -la'): {score('ls -la', 'ls  -la')}")
    print(f"score('ls -la', 'ls -l -a'): {score('ls -la', 'ls -l -a')}")
    
    print("\n=== Tier 3: Semantic Match (requires LLM) ===")
    print("Run with OPENROUTER_API_KEY set to test semantic matching")
    
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        import dspy
        
        judge_lm = dspy.LM(
            model="openrouter/anthropic/claude-sonnet-4",
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
        )
        
        # Test semantic equivalence
        result = semantic_match(
            input_text="show first 20 lines of file.txt",
            expected="head -20 file.txt",
            predicted="head -n 20 file.txt",
            judge_lm=judge_lm,
        )
        print(f"'head -20 file.txt' vs 'head -n 20 file.txt': {result}")
        
        result = semantic_match(
            input_text="list files",
            expected="ls",
            predicted="echo hello",
            judge_lm=judge_lm,
        )
        print(f"'ls' vs 'echo hello': {result}")
