"""Module entrypoint for ``python3 -m aira``."""

from __future__ import annotations

from aira.cli import main_entry


if __name__ == "__main__":
    raise SystemExit(main_entry())
