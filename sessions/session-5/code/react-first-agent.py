"""ReAct in one file (§7): an agent from the module zoo.

dspy.ReAct turns plain Python functions into tools and loops
thought -> tool call -> observation until it can answer. Underneath it is
just more modules: a Predict driving the loop, a ChainOfThought extracting
the final answer.
"""

import math

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=800, temperature=0.0))

ROOMS = {"auditorium": 120, "classroom_a": 30, "classroom_b": 24, "lobby": 15}


# Plain functions are enough: ReAct wraps each in dspy.Tool, reading the
# name, type hints, and docstring to describe it to the LM.
def room_capacity(room: str) -> str:
    """Look up how many people a PySprings meetup room holds."""
    if room.lower() not in ROOMS:
        return f"Unknown room {room!r}. Known rooms: {sorted(ROOMS)}"
    return f"{ROOMS[room.lower()]} people"


def pizzas_needed(people: int, slices_each: int = 3) -> int:
    """Compute how many 8-slice pizzas feed this many people."""
    return math.ceil(people * slices_each / 8)


agent = dspy.ReAct("question -> answer", tools=[room_capacity, pizzas_needed], max_iters=5)

question = (
    "We booked the auditorium for the DSPy session and expect it to be "
    "half full. How many pizzas should we order?"
)
result = agent(question=question)
print(f"question : {question}")
print(f"answer   : {result.answer}")

# The trajectory is a flat dict recording each loop turn: thought_i says
# why, tool_name_i/tool_args_i say what, observation_i is what came back.
print("\n--- the trajectory ---")
for key, value in result.trajectory.items():
    text = str(value)
    if len(text) > 80:
        text = text[:80] + "..."
    print(f"{key:14s}: {text}")

# Getting to 23 pizzas required chaining both tools: capacity first,
# halve it, then feed 60 into the pizza math. The LM planned that.
assert "23" in str(result.answer), result.answer

# No new machinery here: ReAct is a Predict running the tool loop plus a
# ChainOfThought extracting the answer from the finished trajectory.
# Modules all the way down, which is why Session 7's optimizers can tune
# an agent exactly like any other program.
print("\n--- ReAct is modules composed ---")
print([name for name, _ in agent.named_predictors()])

# Uncomment to see the full compiled prompt + raw LM response for the last call:
# dspy.inspect_history(n=1)
