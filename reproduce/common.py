"""Shared runner for the reproduction scripts.

A Step compiles or runs one thing and says how to tell it is done (its output exists).
run_steps runs the steps in order: a done step is skipped unless --force, a failing step
stops the run (the command's exit status is checked). With --force a step first deletes
what it produced (`clean`), so it starts from nothing: a whole file, or (file, key) for
one entry of a JSON file that other scripts also write into. check_rows compares the
regenerated values with the numbers the paper prints and writes them to a CSV; it exits
nonzero if any row differs.
"""
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "reproduce" / "results"


@dataclass
class Step:
    name: str
    cmd: list[str]
    cwd: Path
    done: Callable[[], bool]
    env: dict[str, str]
    clean: list[Path | tuple[Path, str]]   # what --force deletes first: a file, or (file, key)


def json_has(path: Path, *keys: str) -> Callable[[], bool]:
    """done-check: the JSON file exists and has every one of `keys` at its top level."""
    def check() -> bool:
        if not path.exists():
            return False
        doc = json.loads(path.read_text())
        return all(k in doc for k in keys)
    return check


def file_exists(path: Path) -> Callable[[], bool]:
    return path.exists


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def remove(item: Path | tuple[Path, str]) -> None:
    """Deletes a file, or one top-level key of a JSON file (the file's other keys stay)."""
    if isinstance(item, tuple):
        path, key = item
        if path.exists():
            doc = json.loads(path.read_text())
            if key in doc:
                del doc[key]
                path.write_text(json.dumps(doc, indent=2) + "\n")
    elif item.exists():
        item.unlink()


def run_steps(steps: list[Step], force: bool) -> list[dict]:
    """Runs the steps in order; returns one {step, status, seconds} per step."""
    log = []
    for step in steps:
        if not force and step.done():
            print(f"[skip] {step.name}: output exists", flush=True)
            log.append({"step": step.name, "status": "skipped", "seconds": 0.0})
            continue
        if force:
            for item in step.clean:
                remove(item)
        print(f"[run ] {step.name}: {' '.join(step.cmd)}", flush=True)
        t0 = time.time()
        subprocess.run(step.cmd, cwd=step.cwd, env={**os.environ, **step.env}, check=True)
        seconds = time.time() - t0
        if not step.done():
            raise RuntimeError(f"{step.name} finished but its output is missing")
        print(f"[done] {step.name} ({seconds:.0f} s)", flush=True)
        log.append({"step": step.name, "status": "ran", "seconds": round(seconds, 1)})
    return log


def check_rows(name: str, rows: list[dict], log: list[dict]) -> None:
    """Writes RESULTS/<name>.csv (one row per paper value) and RESULTS/<name>_steps.csv
    (the run log), prints the comparison, and exits 1 if any row differs from the paper.

    Each row: table, quantity, model (float), reported (float, or None where the source
    paper gives no value, printed "---"), model_decimals and reported_decimals (the
    paper's printed precision of each), paper_model, paper_reported (the strings the
    paper prints)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out, bad = [], []
    for r in rows:
        model = f"{r['model']:.{r['model_decimals']}f}"
        reported = "---" if r["reported"] is None else f"{r['reported']:.{r['reported_decimals']}f}"
        ok = model == r["paper_model"] and reported == r["paper_reported"]
        out.append({"table": r["table"], "quantity": r["quantity"], "model": model, "reported": reported,
                    "paper_model": r["paper_model"], "paper_reported": r["paper_reported"],
                    "status": "match" if ok else "DIFFERS"})
        if not ok:
            bad.append(out[-1])
    path = RESULTS / f"{name}.csv"
    write_csv(path, out)
    write_csv(RESULTS / f"{name}_steps.csv", log)
    for o in out:
        print(f"  {o['status']:8s} {o['table']:12s} {o['quantity']:40s} model {o['model']:>9s} "
              f"(paper {o['paper_model']:>9s})  reported {o['reported']:>9s} (paper {o['paper_reported']:>9s})")
    print(f"wrote {path}")
    if bad:
        print(f"{len(bad)} of {len(out)} rows differ from the paper", file=sys.stderr)
        sys.exit(1)
    print(f"all {len(out)} rows match the paper")
