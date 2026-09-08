"""GEPA on the Unix task (§5): reflective search over the instruction.

GEPA runs the program on a few training examples, hands the traces and the
metric's feedback text to a reflection model, and asks it to write a better
instruction for one predictor. Candidates that score best on at least one
validation example stay in the pool. It writes instructions only; the demos
slot stays empty.

The metric has GEPA's five-argument shape and returns a score with a
feedback sentence. See common.command_feedback.
"""

import dspy

from common import (JUDGE_LM, REFLECTION_LM, STATE, TASK_LM, UnixAssistant,
                    command_feedback, command_match, cost, load_unix)

dspy.configure(lm=TASK_LM)

trainset, valset, testset = load_unix("train"), load_unix("val"), load_unix("test")
program = UnixAssistant()
before = program.predict.signature.instructions

optimizer = dspy.GEPA(
    metric=command_feedback,
    auto="light",
    reflection_lm=REFLECTION_LM,
    num_threads=6,
    track_stats=True,
)
optimized = optimizer.compile(program, trainset=trainset, valset=valset)

print("--- instruction, before ---")
print(before)
print("\n--- instruction, after ---")
print(optimized.predict.signature.instructions)
print(f"\ndemos after : {len(optimized.predict.demos)}")

print("\n--- search statistics ---")
stats = optimized.detailed_results
print(f"candidates tried   : {len(stats.candidates)}")
print(f"metric calls       : {stats.total_metric_calls}")
print(f"full val evals     : {stats.num_full_val_evals}")
print(f"val score by cand. : {[round(s, 3) for s in stats.val_aggregate_scores]}")
print(f"best candidate     : #{stats.best_idx}")

print("\n--- test split, before and after ---")
evaluator = dspy.Evaluate(devset=testset, metric=command_match, num_threads=4)
print(f"baseline : {evaluator(UnixAssistant()).score}")
print(f"gepa     : {evaluator(optimized).score}")

STATE.mkdir(exist_ok=True)
optimized.save(STATE / "unix-gepa.json")
print(f"\nsaved to {STATE.name}/unix-gepa.json")
print(f"cost     : task ${cost(TASK_LM):.3f}  judge ${cost(JUDGE_LM):.3f}  "
      f"reflection ${cost(REFLECTION_LM):.3f}")
