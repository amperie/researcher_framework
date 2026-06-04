"""Framework-owned external tasks reusable across profiles."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

from core.utils.temp_paths import temporary_directory


def run_tests(payload: dict[str, Any]) -> dict[str, Any]:
    timeout = int(payload.get("timeout") or 0)
    staged_files = dict(payload.get("staged_files") or {})
    task_env = {str(k): str(v) for k, v in dict(payload.get("env") or {}).items()}
    base_cwd = payload.get("cwd") or None
    cmd = [str(part) for part in (payload.get("cmd") or []) if str(part)]

    if staged_files:
        with temporary_directory(prefix="rf_validate_", category="validation") as tmpdir:
            root = Path(tmpdir)
            for rel_path, source in staged_files.items():
                path = root / str(rel_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(source), encoding="utf-8")

            test_path = str(payload.get("test_path") or "")
            if test_path:
                cmd = [str(part) for part in (payload.get("cmd_prefix") or []) if str(part)]
                cmd += [str((root / test_path).resolve())]
                cmd += [str(part) for part in (payload.get("cmd_suffix") or []) if str(part)]

            env = os.environ.copy()
            env.update(task_env)
            for key in ("GENERATED_SCRIPT_PATH", "VALIDATION_PLATFORM_ROOT"):
                if key in env:
                    candidate = Path(env[key])
                    if not candidate.is_absolute():
                        env[key] = str((root / candidate).resolve())
            # Add tmpdir to PYTHONPATH so staged files (generated impl) are importable
            # when running in base_cwd (the plugin project dir) for correct venv resolution.
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
            run_cwd = base_cwd if base_cwd else str(root)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout if timeout > 0 else None,
                    cwd=run_cwd,
                    env=env,
                    check=False,
                )
                return {"output": (result.stdout or "") + (result.stderr or "")}
            except subprocess.TimeoutExpired:
                return {"output": f"TIMEOUT: tests exceeded {timeout}s"}
            except Exception as exc:
                return {"output": f"ERROR running tests: {exc}"}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout if timeout > 0 else None,
            cwd=base_cwd,
            check=False,
        )
        return {"output": (result.stdout or "") + (result.stderr or "")}
    except subprocess.TimeoutExpired:
        return {"output": f"TIMEOUT: tests exceeded {timeout}s"}
    except Exception as exc:
        return {"output": f"ERROR running tests: {exc}"}
