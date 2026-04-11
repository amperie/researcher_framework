# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`neuralsignalresearcher` is a Python 3.12 project managed with [uv](https://github.com/astral-sh/uv). It is currently in early development — `main.py` is the sole entry point.

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py

# Add a dependency
uv add <package>
```

## Structure

- `main.py` — application entry point
- `pyproject.toml` — project metadata and dependencies (uv-managed)
- `.python-version` — pins Python 3.12
- `.venv/` — local virtual environment (gitignored)
