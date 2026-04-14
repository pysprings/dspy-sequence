"""Generate docs/index.html listing every built presentation.

Scans the output directory for presentation HTML files and writes a simple
landing page grouping session decks and case-study tours. Invoked by the
Makefile after all other build targets succeed.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


SESSION_RE = re.compile(r"session-(\d+)")


def session_sort_key(path: Path) -> tuple[int, str]:
    m = SESSION_RE.search(path.as_posix())
    return (int(m.group(1)) if m else 10**9, path.as_posix())


def collect(root: Path) -> tuple[list[Path], list[Path]]:
    sessions = sorted(root.glob("session-*/presentation.html"), key=session_sort_key)
    case_studies = sorted(root.glob("case-studies/*/tour.html"))
    return sessions, case_studies


def render_list(root: Path, paths: list[Path]) -> str:
    items = []
    for p in paths:
        href = p.relative_to(root).as_posix()
        label = html.escape(p.parent.name.replace("-", " ").title())
        items.append(f'    <li><a href="{html.escape(href)}">{label}</a></li>')
    return "\n".join(items) if items else "    <li><em>none built yet</em></li>"


def render(root: Path) -> str:
    sessions, case_studies = collect(root)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DSPy Mastery Series</title>
<style>
  body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ margin-bottom: 0.25rem; }}
  h2 {{ margin-top: 2rem; }}
  a {{ color: #0366d6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  ul {{ padding-left: 1.25rem; }}
</style>
</head>
<body>
<h1>DSPy Mastery Series</h1>
<p>Presentation materials for the PySprings DSPy Mastery Series.</p>

<h2>Sessions</h2>
<ul>
{render_list(root, sessions)}
</ul>

<h2>Case Studies</h2>
<ul>
{render_list(root, case_studies)}
</ul>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="docs", type=Path)
    args = parser.parse_args()

    out = args.root / "index.html"
    out.write_text(render(args.root), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
