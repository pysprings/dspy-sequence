"""Code execution (§4): ProgramOfThought against ChainOfThought.

ProgramOfThought asks the model to write a Python program, runs it, and
hands the program plus its output to a final predictor. The run happens in
a CodeInterpreter. DSPy ships one backed by Deno and Pyodide; this session
uses exec_interpreter.ExecInterpreter, forty lines on exec, so nothing has
to be installed. It is not a sandbox. See the module docstring.

The question is the one from the DSPy docs: a factorial divided by a sum of
primes. Language models drift on long arithmetic; interpreters do not.

The interpreter is created here and passed in, the caller-owned path, so
the script can read back every program the model wrote from interp.log.
"""

import re

import dspy

from common import TASK_LM, cost
from exec_interpreter import ExecInterpreter

dspy.configure(lm=TASK_LM)

question = ("Compute 12! divided by the sum of the prime numbers between 1 and 30. "
            "Give the result to three decimal places.")
expected = 479001600 / 129   # 3713190.698

cot = dspy.ChainOfThought("question -> answer")
pot = dspy.ProgramOfThought("question -> answer", max_iters=3)

print("--- what ProgramOfThought is made of ---")
print([name for name, _ in pot.named_predictors()])

print("\n--- chain of thought ---")
c = cot(question=question)
print(f"answer : {c.answer}")

print("\n--- program of thought ---")
interp = ExecInterpreter()
p = pot(interp, question=question)
interp.shutdown()
print(f"answer : {p.answer}")

print(f"\n--- the program(s) it wrote: {len(interp.log)} attempt(s) ---")
for i, code in enumerate(interp.log, 1):
    print(f"[attempt {i}]")
    print("\n".join("    " + line for line in code.splitlines() if line.strip()))

print(f"\nexpected : {expected:.3f}")
print(f"cost     : ${cost(TASK_LM):.4f}")

def close(text: str) -> bool:
    """True if any number in the text is within a cent of the expected value."""
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", str(text))
    return any(abs(float(n.replace(",", "")) - expected) < 0.01 for n in numbers)

assert close(p.answer), p.answer
print(f"cot close to expected : {close(c.answer)}")

# Uncomment to see the generate prompt, with the SUBMIT() instruction:
# pot.code_generate.inspect_history(n=1)
