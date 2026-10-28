"""Run one dark-deploy workflow step in bash against stubbed Railway and HTTP.

Only the external CLIs (``railway``, ``curl``, ``git``, ``date``, ``sleep``)
are stubbed. ``jq`` and bash are real, so the step's own parsing and control
flow are what the tests exercise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "dark-deploy.yml"

_STUB = r'''#!PYTHON
import json, os, sys
name = os.path.basename(sys.argv[0])
args = sys.argv[1:]
scenario = json.load(open(os.environ["STUB_SCENARIO"]))
state_path = os.environ["STUB_STATE"]
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
entry = {"tool": name, "args": args}
if name == "curl" and "--data-binary" in args:
    entry["data"] = open(args[args.index("--data-binary") + 1].lstrip("@")).read()
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps(entry) + "\n")
joined = " ".join(args)
for index, rule in enumerate(scenario.get(name, [])):
    if rule["match"] not in joined:
        continue
    if "data_contains" in rule and rule["data_contains"] not in entry.get("data", ""):
        continue
    key = f"{name}:{index}"
    count = state.get(key, 0)
    state[key] = count + 1
    json.dump(state, open(state_path, "w"))
    responses = rule["responses"]
    response = responses[min(count, len(responses) - 1)]
    body = response.get("body", "")
    text = body if isinstance(body, str) else json.dumps(body)
    status = response.get("status", 200)
    if name == "curl":
        if "--fail" in args and status >= 400:
            sys.stderr.write(f"curl: (22) The requested URL returned error: {status}\n")
            sys.exit(22)
        if "--output" in args:
            open(args[args.index("--output") + 1], "w").write(text)
        else:
            sys.stdout.write(text)
        if "--write-out" in args:
            sys.stdout.write(str(status))
    else:
        sys.stdout.write(text)
    sys.exit(response.get("exit", 0))
if name == "sleep":
    sys.exit(0)
sys.stderr.write(f"unstubbed {name} call: {joined}\n")
sys.exit(97)
'''


def step(name: str) -> dict[str, Any]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for candidate in workflow["jobs"]["dark-deploy"]["steps"]:
        if candidate.get("name") == name:
            return candidate
    raise AssertionError(f"no step named {name!r}")


def run_step(
    tmp_path: Path,
    name: str,
    *,
    env: dict[str, str],
    scenario: dict[str, list[dict[str, Any]]],
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]], dict[str, str]]:
    script = step(name)["run"]
    assert "${{" not in script, "pass workflow expressions through step env"
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    script = script.replace("/tmp/", f"{work}/")
    (tmp_path / "ops").mkdir(exist_ok=True)
    for query in (ROOT / "ops").glob("*.graphql"):
        (tmp_path / "ops" / query.name).write_text(query.read_text(encoding="utf-8"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in {"railway", "curl", "sleep", *scenario}:
        stub = bin_dir / tool
        stub.write_text(_STUB.replace("PYTHON", sys.executable, 1), encoding="utf-8")
        stub.chmod(0o755)
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    state_path = tmp_path / "stub_state.json"
    if state_path.exists():
        state_path.unlink()
    output_path = tmp_path / "github_output"
    output_path.write_text("", encoding="utf-8")
    log_path = tmp_path / "calls.jsonl"
    if log_path.exists():
        log_path.unlink()
    full_env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "STUB_SCENARIO": str(scenario_path),
        "STUB_STATE": str(state_path),
        "STUB_LOG": str(log_path),
        "GITHUB_OUTPUT": str(output_path),
        **env,
    }
    # GitHub runs `shell: bash` steps as `bash --noprofile --norc -eo pipefail`.
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=tmp_path,
        env=full_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    calls = [
        json.loads(line)
        for line in (log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else [])
    ]
    outputs = dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    return result, calls, outputs


def deployments(*rows: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": deployment_id,
            "status": status,
            "createdAt": f"2026-10-26T17:{59 - index:02d}:00Z",
            "meta": {"cliMessage": f"dark release for {deployment_id}"},
        }
        for index, (deployment_id, status) in enumerate(rows)
    ]
