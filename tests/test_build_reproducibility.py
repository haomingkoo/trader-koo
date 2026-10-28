"""Static contracts for the reproducible Railway and CI build inputs.

These read the committed build files only. They prove the build is wired to
the hashed lock and pinned toolchains; they cannot prove a Railway build
succeeds, which only the first dark deploy shows.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
LOCK = ROOT / "requirements.txt"
DEV_LOCK = ROOT / "requirements-dev.txt"
LOCK_INPUT = ROOT / "requirements.in"
DEV_LOCK_INPUT = ROOT / "requirements-dev.in"
NVMRC = ROOT / "trader_koo" / "frontend-v2" / ".nvmrc"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+?)(?:\s*;.*?)?\s*\\?$")


def _locked_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _requirement_blocks(path: Path) -> list[str]:
    """Join backslash-continued requirement lines into one string each."""
    blocks: list[str] = []
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not current and (not stripped or stripped.startswith(("#", "-"))):
            continue
        current += " " + stripped.rstrip("\\")
        if not stripped.endswith("\\"):
            blocks.append(current.strip())
            current = ""
    return blocks


def test_every_locked_requirement_is_pinned_with_a_hash() -> None:
    for path in (LOCK, DEV_LOCK):
        blocks = _requirement_blocks(path)
        assert blocks, path.name
        for block in blocks:
            assert "==" in block and "--hash=sha256:" in block, block


def test_runtime_lock_resolves_cpu_torch_without_cuda() -> None:
    pins = _locked_pins(LOCK)
    text = LOCK.read_text(encoding="utf-8")

    assert f"--extra-index-url {TORCH_CPU_INDEX}" in text
    assert pins["torch"] == "2.13.0+cpu"
    assert pins["torchvision"] == "0.28.0+cpu"
    assert pins["opencv-python-headless"] == "4.10.0.84"
    assert not [name for name in pins if name.startswith("nvidia-")]
    assert "triton" not in pins
    # torch needs setuptools; Dependabot adds --allow-unsafe when it sees this
    # pin, so the lock must be compiled with it too or updates would diverge.
    assert "setuptools" in pins
    assert "--allow-unsafe" in text.split("\n\n", 1)[0]


def test_lock_input_selects_cpu_torch_on_linux_and_plain_wheels_elsewhere() -> None:
    text = LOCK_INPUT.read_text(encoding="utf-8")

    assert f"--extra-index-url {TORCH_CPU_INDEX}" in text
    assert 'torch==2.13.0+cpu ; sys_platform == "linux"' in text
    assert 'torch==2.13.0 ; sys_platform != "linux"' in text
    assert 'torchvision==0.28.0+cpu ; sys_platform == "linux"' in text
    assert 'torchvision==0.28.0 ; sys_platform != "linux"' in text


def test_dev_lock_contains_every_runtime_pin() -> None:
    # Dependabot recompiles only the .in files whose lock already names the
    # bumped package, and it follows -r but not -c. Including requirements.in
    # puts every runtime package in the dev lock, so a runtime bump always
    # regenerates both locks; -c keeps the dev lock on the runtime versions.
    lines = DEV_LOCK_INPUT.read_text(encoding="utf-8").splitlines()
    assert "-r requirements.in" in lines
    assert "-c requirements.txt" in lines
    runtime = _locked_pins(LOCK)
    dev = _locked_pins(DEV_LOCK)
    assert {name: dev.get(name) for name in runtime} == runtime
    assert {"pytest", "pylint", "import-linter"} <= dev.keys()
    assert f"--extra-index-url {TORCH_CPU_INDEX}" in DEV_LOCK.read_text(encoding="utf-8")


def test_nixpacks_installs_only_the_hashed_lock() -> None:
    phases = tomllib.loads((ROOT / "nixpacks.toml").read_text(encoding="utf-8"))["phases"]
    (install,) = phases["install"]["cmds"]

    assert "python -m venv --copies /opt/venv" in install
    assert "pip install --no-deps --require-hashes -r requirements.txt" in install
    assert "/opt/venv/bin" in phases["install"]["paths"]


def test_dependabot_keeps_torch_pins_manual() -> None:
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    (root_pip,) = [
        update
        for update in config["updates"]
        if update["package-ecosystem"] == "pip" and update["directory"] == "/"
    ]
    ignored = {entry["dependency-name"] for entry in root_pip.get("ignore", [])}
    assert {"torch", "torchvision"} <= ignored


def test_ci_checks_the_locks_match_their_inputs() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    lock_job = workflow["jobs"]["dependency-lock"]
    runs = [step.get("run", "") for step in lock_job["steps"]]

    assert any("make lock-check" in run for run in runs)
    for step in lock_job["steps"]:
        uses = step.get("uses")
        if uses:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", uses), uses


def test_nixpacks_pins_python_and_node_to_one_revision() -> None:
    phases = tomllib.loads((ROOT / "nixpacks.toml").read_text(encoding="utf-8"))["phases"]

    # Python and Node share one revision; see the comment in nixpacks.toml.
    assert re.fullmatch(r"[0-9a-f]{40}", phases["setup"]["nixpkgsArchive"])
    assert {"python311", "nodejs_22"} <= set(phases["setup"]["nixPkgs"])
    assert [name for name, phase in phases.items() if "nixpkgsArchive" in phase] == ["setup"]


def test_node_version_is_exact_and_shared_with_ci() -> None:
    version = NVMRC.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"22\.\d+\.\d+", version)

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    node_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/setup-node@")
    ]
    assert node_steps
    for step in node_steps:
        assert step["with"].get("node-version-file") == "trader_koo/frontend-v2/.nvmrc"
        assert "node-version" not in step["with"]
        assert re.fullmatch(r"actions/setup-node@[0-9a-f]{40}", step["uses"])
