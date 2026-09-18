"""Start the researcher playground: python scripts/start-playground.py [--port 8091]."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8091)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--no-reload", action="store_true", help="Disable the development reload subprocess")
parser.add_argument("--model", default=os.environ.get("RESEARCHER_LLM_MODEL", "claude-haiku-4-5-20251001"))
args = parser.parse_args()
python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
env = {**os.environ, "RESEARCHER_LLM_PROVIDER": "anthropic", "RESEARCHER_LLM_MODEL": args.model}
if not python.exists():
    if not shutil.which("uv"):
        parser.error("Install uv or create the repository's .venv first")
    env["UV_PROJECT_ENVIRONMENT"] = str(root / ".tmp/playground-venv")
    subprocess.run(["uv", "sync", "--frozen", "--only-group", "platform", "--no-install-project"], cwd=root, env=env, check=True)
    python = Path(env["UV_PROJECT_ENVIRONMENT"]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
print(f"Researcher playground: http://{args.host}:{args.port} | Claude: {args.model}", flush=True)
try:
    raise SystemExit(subprocess.call([str(python), "-m", "uvicorn", "core.platform.playground:create_playground",
        "--factory", *([] if args.no_reload else ["--reload"]), "--host", args.host, "--port", str(args.port)], cwd=root, env=env))
except KeyboardInterrupt:
    pass
