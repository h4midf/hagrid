"""report_code_fingerprint: the code fingerprint the resumable sweeps store with their
results."""
import hashlib


def report_code_fingerprint(*extra_paths: str) -> str:
    """Truncated SHA-256 of this module's source plus `extra_paths` (e.g. the caller's
    report file). Resumable sweeps store it with their results and discard every cached
    entry when it changes, so results computed with older formulas are never reused."""
    h = hashlib.sha256()
    h.update(open(__file__, "rb").read())
    for p in extra_paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]
