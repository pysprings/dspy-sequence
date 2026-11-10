"""CodeAct (§4): tools called from generated code.

Session 5's ReAct agent called room_capacity and pizzas_needed one JSON tool
call at a time. CodeAct gives the model the same tools as Python functions
and asks for a program instead. One iteration can call both.

Two things the run shows. CodeAct defines each tool inside the interpreter
by sending its source, so a tool must be a plain function that carries its
own imports; that is why pizzas_needed imports math inside its body. And
the task model matters more here than in ReAct: gpt-4.1-nano reads the tool
schema as a dict argument and never recovers, while gpt-4.1-mini writes
keyword calls and finishes in one iteration.
"""

import logging

import dspy

from common import JUDGE_LM, TASK_LM, cost
from exec_interpreter import ExecInterpreter

# CodeAct passes its trajectory dict straight to the predictor, where ReAct
# formats it to a string first; DSPy 3.3.0 warns about the type on every
# turn. The warning is noise for this demo.
logging.getLogger("dspy.predict.predict").setLevel(logging.ERROR)

dspy.configure(lm=TASK_LM)


def room_capacity(room: str) -> int:
    """Look up how many people a PySprings meetup room holds."""
    rooms = {"auditorium": 120, "classroom_a": 30, "classroom_b": 24, "lobby": 15}
    return rooms[room.lower()]


def pizzas_needed(people: int, slices_each: int = 3) -> int:
    """Compute how many 8-slice pizzas feed this many people."""
    import math
    return math.ceil(people * slices_each / 8)


question = (
    "We booked the auditorium for the DSPy session and expect it to be "
    "half full. How many pizzas should we order?"
)

print("--- ReAct, as in Session 5 (gpt-4.1-nano) ---")
react = dspy.ReAct("question -> answer", tools=[room_capacity, pizzas_needed], max_iters=5)
r = react(question=question)
print(f"answer      : {r.answer}")
print(f"tool calls  : {[v for k, v in r.trajectory.items() if k.startswith('tool_name')]}")
assert "23" in str(r.answer), r.answer

act = dspy.CodeAct("question -> answer", tools=[room_capacity, pizzas_needed], max_iters=5,
                   interpreter_factory=ExecInterpreter)

for label, lm in [("gpt-4.1-nano", TASK_LM), ("gpt-4.1-mini", JUDGE_LM)]:
    print(f"\n--- CodeAct ({label}) ---")
    with dspy.context(lm=lm):
        a = act(question=question)
    print(f"answer      : {a.answer}")
    codes = [k for k in a.trajectory if k.startswith("generated_code")]
    print(f"iterations  : {len(codes)}")
    first = a.trajectory[codes[0]]
    print("first program:")
    print("\n".join("    " + line for line in first.splitlines() if line.strip() and not line.strip().startswith("#")))
    obs = a.trajectory.get("observation_0") or a.trajectory.get("code_output_0")
    print(f"first result : {str(obs).strip()[:120]}")

assert "23" in str(a.answer), a.answer   # the mini run, last in the loop

print("\n--- both are modules composed ---")
print(f"ReAct   : {[name for name, _ in react.named_predictors()]}")
print(f"CodeAct : {[name for name, _ in act.named_predictors()]}")
print(f"\ncost : nano ${cost(TASK_LM):.4f}  mini ${cost(JUDGE_LM):.4f}")
