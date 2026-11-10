"""Throughput (§6): the same 48 calls three ways.

A plain loop, Module.batch (threads), and asyncio.gather over acall. The
cost is the same; the wall time is not. The LM here is built with
cache=False, because with the default cache the second and third methods
would hit cached completions and finish instantly, which would prove
nothing about threads or the event loop.

Then dspy.Parallel on its own, the thing batch and Evaluate are built on,
and the grounded inbox's aforward, a custom module on the async path.
"""

import asyncio
import statistics
import time

import dspy

from common import GroundedInbox, UnixAssistant, cost, load_unix

NO_CACHE_LM = dspy.LM("openrouter/openai/gpt-4.1-nano", max_tokens=400, cache=False)
dspy.configure(lm=NO_CACHE_LM)

rows = load_unix()          # all 48 rows: train, val, and test
assistant = UnixAssistant()
print(f"--- {len(rows)} requests, three ways, three rounds each ---")


def run_loop():
    return [assistant(request=r.request) for r in rows]


def run_batch():
    return assistant.batch(rows, num_threads=8, disable_progress_bar=True)


async def _gather():
    return await asyncio.gather(*(assistant.acall(request=r.request) for r in rows))


def run_async():
    return asyncio.run(_gather())


timings = {}
for label, fn in [("loop", run_loop), ("batch", run_batch), ("async", run_async)]:
    samples = []
    for _ in range(3):
        before = cost(NO_CACHE_LM)
        start = time.perf_counter()
        results = fn()
        samples.append(time.perf_counter() - start)
        assert len(results) == len(rows)
    timings[label] = (statistics.median(samples), cost(NO_CACHE_LM) - before)

print(f"\n{'method':8s} {'median s':>9s} {'per call ms':>12s} {'cost/round':>11s}")
for label, (seconds, spent) in timings.items():
    print(f"{label:8s} {seconds:>9.2f} {1000 * seconds / len(rows):>12.0f} {spent:>11.4f}")
assert timings["batch"][0] < timings["loop"][0]
assert timings["async"][0] < timings["loop"][0]

print("\n--- dspy.Parallel directly: (module, example) pairs ---")
pairs = [(assistant, r) for r in rows[:4]]
parallel = dspy.Parallel(num_threads=4, disable_progress_bar=True)
for r, out in zip(rows[:4], parallel(pairs)):
    print(f"{r.request[:44]:44s} -> {out.command}")

print("\n--- a custom module on the async path: GroundedInbox.aforward ---")
grounded = GroundedInbox(k=3)
emails = [
    "I was charged twice for my March invoice. Please refund the duplicate.",
    "Minor thing: the settings page misspells 'preferences' in the sidebar.",
]


async def drafts():
    return await asyncio.gather(*(grounded.acall(email=e) for e in emails))


for e, out in zip(emails, asyncio.run(drafts())):
    print(f"{e[:50]:50s} -> {out.category}/{out.urgency}: {out.reply[:70]}...")

print(f"\ncost : ${cost(NO_CACHE_LM):.4f}")
