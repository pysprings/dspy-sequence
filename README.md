# DSPy Mastery Series

This repository contains the presentation materials for the PySprings "DSPy
Mastery Series," a 12-month structured exploration of the DSPy framework.

## About the Series

We're launching a structured 12-month exploration of DSPy, the framework that
applies familiar Python patterns to AI systems. Instead of manual prompt
engineering, DSPy enables declarative programming where you specify what you
want and the system automatically optimizes how to achieve it.

This curriculum is being developed collaboratively—your feedback, questions,
and real-world applications will shape the content as we pioneer this learning
path together.

## Schedule

The series runs on the 2nd Tuesday of each month.

- **Session 0:** Introduction - AI Development the Python Way
- **Session 1:** LM Setup - Gateway to AI Services
- **Session 2:** Data Collection - Building Training Foundations
- **Session 3:** Signatures - Declaring What You Want
- **Session 4:** Adapters - The Translation Layer
- **Session 5:** Basic Modules - Working AI Programs
- **Session 6:** Metrics - Measuring Success
- **Session 7:** Optimization - The DSPy Superpower
- **Session 8:** Advanced Modules - Sophisticated Patterns
- **Session 9:** Assertions - Reliable AI Systems
- **Session 10:** Trackers - Production Observability
- **Session 11:** Retrospective - Mastery Achieved

## Case Studies

In addition to the monthly sessions, we have in-depth case studies that
demonstrate DSPy concepts applied to real-world projects:

- **[Sline](case-studies/sline/TOUR.md):** A shell command assistant with
  DSPy-powered prompt optimization. This case study covers signatures, custom
  adapters, metrics, benchmarking, and MIPROv2 optimization.

## Project Structure

Each session's materials are located in a corresponding directory. The main
presentation file for each session is `presentation.md`. Here's the directory
structure:

```
.
├── Makefile
├── README.md
├── sessions/
│   ├── session-0/
│   │   ├── presentation.md
│   │   └── code/
│   ├── session-1/
│   │   ├── presentation.md
│   │   └── code/
│   ...
│   └── session-11/
│       ├── presentation.md
│       └── code/
└── case-studies/
    └── sline/
        ├── TOUR.md
        └── code/
```

## Building

Build all presentations and case studies to HTML:

```bash
make
```

Output is written to `docs/`.

## Serving over the LAN

To view the slides from another machine on the local network, start a
live-reloading HTTP server:

```bash
make serve
```

This builds everything (if needed), binds to `0.0.0.0:8000`, and watches the
presentation sources. Edits to `sessions/*/presentation.md`,
`case-studies/*/TOUR.md`, `assets/css/*.css`, `assets/images/*.tex`, or the
`Makefile` trigger an incremental rebuild and any connected browsers refresh
automatically.

Connect from another machine at `http://<host>:8000/`. Override the bind
address or port with:

```bash
make serve HOST=127.0.0.1 PORT=9000
```

Requires `livereload` in the project virtualenv (`uv pip install livereload`).

