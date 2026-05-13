"""DSPy MIPROv2 optimization for sline prompt.

Demonstrates the complete optimization workflow:
1. Load and augment training examples
2. Configure multi-model setup (inference, teacher, judge)
3. Run MIPROv2 optimization
4. Extract and save the optimized prompt

This is a simplified version of the full sline_harness.optimize module.
"""

import csv
import os
import random
from pathlib import Path

import dspy
from dspy.teleprompt import MIPROv2


# === Signature and Module ===

class ShellAssistant(dspy.Signature):
    """You are a shell command assistant. Given a natural language description 
    or partial command, return ONLY a valid shell command. No explanation, no 
    markdown, no code fences - just the raw command.

    Context:
    - Shell: {shell}
    - OS: {os_name}
    - Working directory: {cwd}

    Rules:
    1. Output exactly one command (may use pipes, &&, etc.)
    2. Use syntax appropriate for the specified shell
    3. Prefer common utilities available on the OS
    4. If the input is already a valid command with a typo, fix it
    5. If unclear, make a reasonable assumption"""
    
    shell: str = dspy.InputField()
    os_name: str = dspy.InputField()
    cwd: str = dspy.InputField()
    request: str = dspy.InputField()
    command: str = dspy.OutputField()


class ShellAssistantModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ShellAssistant)
    
    def forward(self, shell: str, os_name: str, cwd: str, request: str) -> dspy.Prediction:
        return self.predict(shell=shell, os_name=os_name, cwd=cwd, request=request)


# === Data Loading and Augmentation ===

# Fixed context for training
CONTEXT = {
    "shell": "zsh 5.9 (x86_64-ubuntu-linux-gnu)",
    "os_name": "Linux Mint 22.1",
    "cwd": "/home/ryan/src/workshop",
}


def inject_typo(s: str) -> str:
    """Inject a typo via character swap or deletion."""
    if len(s) < 2:
        return s
    i = random.randint(0, len(s) - 2)
    if random.random() < 0.5:
        # Swap adjacent chars
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    else:
        # Delete a char
        return s[:i] + s[i + 1:]


def augment_example(input_text: str, expected: str) -> list[dict]:
    """Generate original + 2 mutations."""
    return [
        {"input": input_text, "expected": expected, "category": "original"},
        {"input": inject_typo(input_text), "expected": expected, "category": "typo"},
        {"input": random.choice([input_text.lower(), input_text.upper()]), "expected": expected, "category": "case"},
    ]


def load_examples(tsv_path: str, augment: bool = True) -> list[dspy.Example]:
    """Load examples from TSV as DSPy Examples."""
    examples = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if augment:
                for aug in augment_example(row["input"], row["expected"]):
                    ex = dspy.Example(
                        shell=CONTEXT["shell"],
                        os_name=CONTEXT["os_name"],
                        cwd=CONTEXT["cwd"],
                        request=aug["input"],
                        command=aug["expected"],
                    ).with_inputs("shell", "os_name", "cwd", "request")
                    examples.append(ex)
            else:
                ex = dspy.Example(
                    shell=CONTEXT["shell"],
                    os_name=CONTEXT["os_name"],
                    cwd=CONTEXT["cwd"],
                    request=row["input"],
                    command=row["expected"],
                ).with_inputs("shell", "os_name", "cwd", "request")
                examples.append(ex)
    return examples


# === Metrics ===

def exact_match(expected: str, predicted: str) -> bool:
    return expected.strip() == predicted.strip()


def semantic_match_simple(request: str, expected: str, predicted: str, judge_lm: dspy.LM) -> bool:
    """Simplified semantic match for demonstration."""
    if exact_match(expected, predicted):
        return True
    
    # LLM judge
    prompt = f"""Are these two shell commands equivalent for: {request}
Expected: {expected}
Predicted: {predicted}
Answer YES or NO only."""
    
    response = judge_lm(prompt)
    answer = str(response[0]).strip().upper()
    return "YES" in answer


def make_metric(judge_lm: dspy.LM):
    """Create metric function for MIPROv2."""
    def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        expected = example.command
        predicted = prediction.command if hasattr(prediction, "command") else ""
        is_match = semantic_match_simple(example.request, expected, predicted, judge_lm)
        return 1.0 if is_match else 0.0
    return metric


# === Main Optimization ===

def main():
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return 1
    
    # Model configuration - three models with different roles
    inference_model = os.environ.get("SLINE_MODEL", "openai/gpt-oss-20b")
    teacher_model = os.environ.get("SLINE_TEACHER_MODEL", "anthropic/claude-opus-4")
    judge_model = os.environ.get("SLINE_JUDGE_MODEL", "anthropic/claude-sonnet-4")
    
    print(f"Inference model: {inference_model}")
    print(f"Teacher model: {teacher_model}")
    print(f"Judge model: {judge_model}")
    
    # Configure DSPy with OpenRouter
    lm = dspy.LM(
        model=f"openrouter/{inference_model}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    
    teacher_lm = dspy.LM(
        model=f"openrouter/{teacher_model}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    
    judge_lm = dspy.LM(
        model=f"openrouter/{judge_model}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    
    dspy.configure(lm=lm)
    
    # Load examples (use a small sample for demonstration)
    # In production, use the full examples.tsv
    sample_examples = [
        dspy.Example(
            shell=CONTEXT["shell"],
            os_name=CONTEXT["os_name"],
            cwd=CONTEXT["cwd"],
            request="list files by size",
            command="ls -lhS",
        ).with_inputs("shell", "os_name", "cwd", "request"),
        dspy.Example(
            shell=CONTEXT["shell"],
            os_name=CONTEXT["os_name"],
            cwd=CONTEXT["cwd"],
            request="gti status",
            command="git status",
        ).with_inputs("shell", "os_name", "cwd", "request"),
        dspy.Example(
            shell=CONTEXT["shell"],
            os_name=CONTEXT["os_name"],
            cwd=CONTEXT["cwd"],
            request="show hidden files",
            command="ls -la",
        ).with_inputs("shell", "os_name", "cwd", "request"),
    ]
    
    print(f"\nUsing {len(sample_examples)} sample examples")
    
    # Create the module
    module = ShellAssistantModule()
    
    # Run MIPROv2 optimization
    print("\nRunning MIPROv2 optimization...")
    print("This may take a few minutes.\n")
    
    optimizer = MIPROv2(
        metric=make_metric(judge_lm),
        prompt_model=teacher_lm,
        task_model=lm,
        auto="light",  # Minimal optimization for quick demo
    )
    
    optimized_module = optimizer.compile(
        module,
        trainset=sample_examples,
        valset=sample_examples,
    )
    
    # Extract the optimized prompt
    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)
    
    # Get the optimized state
    optimized_state = optimized_module.dump_state()
    
    # Extract instructions
    instructions = optimized_state["predict"]["signature"]["instructions"]
    
    # Convert DSPy placeholders to sline's bash format
    instructions = instructions.replace("{shell}", "{{shell}}")
    instructions = instructions.replace("{os_name}", "{{os}}")
    instructions = instructions.replace("{cwd}", "{{cwd}}")
    
    print("\nOptimized Prompt:")
    print("-" * 60)
    print(instructions)
    print("-" * 60)
    
    # Test the optimized module
    print("\nTesting optimized module:")
    for ex in sample_examples:
        result = optimized_module(
            shell=ex.shell,
            os_name=ex.os_name,
            cwd=ex.cwd,
            request=ex.request,
        )
        status = "OK" if exact_match(ex.command, result.command) else "DIFF"
        print(f"  [{status}] '{ex.request}' -> '{result.command}'")
    
    return 0


if __name__ == "__main__":
    exit(main())
