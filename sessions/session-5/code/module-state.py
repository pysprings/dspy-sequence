"""Module state (§8): what saves, what loads, what optimizers write.

A module's learnable state (demos and signature instructions per predictor)
is a plain dict that serializes to JSON and travels. Save it, load it into a
fresh instance of the same architecture, and the learning comes back.
"""

import pathlib
import tempfile
from pprint import pprint

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", max_tokens=200))


class EnglishToUnix(dspy.Signature):
    """Convert natural language requests into Unix commands."""

    request = dspy.InputField(desc="what the user wants to do")
    command = dspy.OutputField(desc="the Unix command to accomplish this")


class UnixAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(EnglishToUnix)

    def forward(self, request: str) -> dspy.Prediction:
        return self.predict(request=request)


assistant = UnixAssistant()

# Hand-attach a demo; this is exactly the slot an optimizer in Session 7
# fills automatically after searching for examples that raise your metric.
assistant.predict.demos = [dspy.Example(request="list files by size", command="ls -lhS")]

print("--- dump_state ---")
state = assistant.dump_state()
# Top-level keys are attribute-path names: one entry per predictor.
print(f"top-level keys : {list(state.keys())}")
print(f"demos          : {state['predict']['demos']}")
print("instructions   :")
pprint(state["predict"]["signature"]["instructions"])

# Roundtrip: state-only save to JSON, then load into a FRESH instance.
# The file holds state, not code; architecture and state must match:
# rename self.predict and the "predict" key in old saves no longer resolves.
print("\n--- save / load roundtrip ---")
with tempfile.TemporaryDirectory() as tmpdir:
    path = pathlib.Path(tmpdir) / "unix-assistant.json"
    assistant.save(path)
    print(f"saved to       : {path.name} ({path.stat().st_size} bytes)")

    fresh = UnixAssistant()
    assert not fresh.predict.demos, fresh.predict.demos
    fresh.load(path)
    assert fresh.predict.demos, "demos did not survive the roundtrip"
    print(f"fresh demos    : {fresh.predict.demos}")

# The loaded module works, and the demo now rides along in every prompt;
# the adapter formats it as a worked example (Session 4).
print("\n--- the loaded module runs ---")
result = fresh(request="show the five largest files in this directory")
print(f"command        : {result.command}")

# Each module keeps its own call log (fed via settings.caller_modules),
# separate from the global dspy.inspect_history.
print("\n--- per-module history ---")
fresh.inspect_history(n=1)
