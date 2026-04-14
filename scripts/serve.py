"""Serve built slides with live rebuild and browser auto-reload.

Runs a local HTTP server rooted at ``docs/``. Watches presentation sources and
styling; when they change, re-runs ``make`` (which rebuilds only stale
artifacts). Injects a WebSocket client into served HTML so any browser viewing
a slide deck refreshes automatically after each rebuild.

Usage:
    python scripts/serve.py [--host HOST] [--port PORT] [--root ROOT]

Defaults bind to 0.0.0.0:8000 for LAN access. Invoke via ``make serve``.
"""

from __future__ import annotations

import argparse

from livereload import Server, shell


WATCHED_SOURCES = (
    "sessions/*/presentation.md",
    "case-studies/*/TOUR.md",
    "assets/css/*.css",
    "assets/images/*.tex",
    "Makefile",
)


def build_server(root: str) -> Server:
    server = Server()
    rebuild = shell("make")
    for pattern in WATCHED_SOURCES:
        server.watch(pattern, rebuild)
    server.watch(f"{root}/**/*.html")
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", default="docs")
    args = parser.parse_args()

    server = build_server(args.root)
    server.serve(
        root=args.root,
        host=args.host,
        port=args.port,
        open_url_delay=None,
    )


if __name__ == "__main__":
    main()
