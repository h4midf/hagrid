"""The data files the paper's plot scripts draw from: one JSON per figure, next to its
output, with everything the plot draws ("data") and how it was made ("meta"), so a
figure can be redrawn from the repo alone. Each plot script writes its file with
--extract (from the sweep results) and draws from it otherwise.

  meta  generated      when the file was written
        git_commit     HEAD when it was written, and whether the tree had changes
        script         the plot script that wrote it
        sources        per results file it read: its path (repo-relative) and the run's
                       own record (target, model, policy, code fingerprint, flags...)
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# the run records copied from a results file into "sources" (those it has)
RUN_KEYS = ("target", "model", "method", "cap", "policy", "code_fingerprint", "flags", "array",
            "min_feasible_array_cap", "max_cap")


def repo_path(path: str) -> Path:
    """A results path as a sweep saved it (repo-relative, or absolute on the machine that
    ran it) as a path in this repo: an absolute one from its run_scripts/ on."""
    p = Path(path)
    if not p.is_absolute():
        return REPO_ROOT / p
    if p.exists():
        return p
    parts = p.parts
    if "run_scripts" not in parts:
        raise FileNotFoundError(f"{path}: not in this repo and not under a run_scripts/ directory")
    return REPO_ROOT.joinpath(*parts[parts.index("run_scripts"):])


def relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def meta(script: str, sources: list[Path]) -> dict:
    """The "meta" of a data file written by `script` (__file__) from the results files `sources`."""
    out = []
    for src in sources:
        record = json.loads(Path(src).read_text())
        out.append({"path": relative(src), **{k: record[k] for k in RUN_KEYS if k in record}})
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "script": relative(Path(script)),
        "sources": out,
    }


def write(path: Path, script: str, sources: list[Path], data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta(script, sources), "data": data}, indent=1, allow_nan=False) + "\n")
    print(f"wrote {path}")


def read(path: Path) -> dict:
    """The "data" of a data file; raises with how to make it if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the plot script with --extract first")
    return json.loads(path.read_text())["data"]
